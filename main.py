import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

from faster_whisper import WhisperModel
from tqdm import tqdm


def transcribe_file(
    model: WhisperModel,
    audio_path: Path,
    output_dir: Path,
    language: Optional[str] = None,
    output_format: str = "txt",
) -> None:
    """
    Распознать один аудиофайл и сохранить результат в output_dir.
    """

    segments, info = model.transcribe(
        str(audio_path),
        language=language,   # можно указать "ru", "en" и т.д.
        vad_filter=True,     # обрезает тишину, ускоряет
        beam_size=5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"

    # Собираем сегменты, чтобы можно было пройтись по ним дважды
    collected: List[Tuple[float, float, str]] = []
    for seg in segments:
        collected.append((seg.start, seg.end, seg.text))

    # Сохраняем обычный текст
    with txt_path.open("w", encoding="utf-8") as f_txt:
        f_txt.write(f"# language={info.language}, duration={info.duration:.1f}s\n\n")
        for _, _, text in collected:
            f_txt.write(text.strip() + "\n")

    # Опционально сохраняем SRT
    if output_format in {"srt", "both"}:
        def format_ts(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t * 1000) % 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        with srt_path.open("w", encoding="utf-8") as f_srt:
            for i, (start, end, text) in enumerate(collected, start=1):
                f_srt.write(f"{i}\n")
                f_srt.write(f"{format_ts(start)} --> {format_ts(end)}\n")
                f_srt.write(text.strip() + "\n\n")


def find_audio_files(path: Path) -> List[Path]:
    """
    Найти все аудиофайлы в пути (если файл — вернуть его, если папка — пройтись рекурсивно).
    """
    exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    if path.is_file():
        return [path] if path.suffix.lower() in exts else []
    files: List[Path] = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Быстрое распознавание больших аудиофайлов с помощью faster-whisper"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Путь к mp3-файлу или папке с файлами",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="transcripts",
        help="Папка, куда сохранять расшифровки (по умолчанию: ./transcripts)",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        type=str,
        default="large-v3",
        help="Размер модели faster-whisper (tiny, base, small, medium, large-v3 и т.д.)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Язык речи (например: ru, en). Если не указать — модель попробует определить сама.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Устройство: cuda или cpu (по умолчанию cuda)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["txt", "srt", "both"],
        default="both",
        help="Формат вывода: только txt, только srt или оба",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise SystemExit(f"Путь не найден: {input_path}")

    print(f"Загружаем модель {args.model_size} на устройстве {args.device}...")
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
    )

    audio_files = find_audio_files(input_path)
    if not audio_files:
        raise SystemExit("Не найдено аудиофайлов (mp3/wav/m4a/ogg/flac).")

    print(f"Найдено файлов: {len(audio_files)}")
    for audio_path in tqdm(audio_files, desc="Распознаём"):
        try:
            transcribe_file(
                model=model,
                audio_path=audio_path,
                output_dir=output_dir,
                language=args.language,
                output_format=args.output_format,
            )
        except Exception as e:
            print(f"Ошибка при обработке {audio_path}: {e}")


if __name__ == "__main__":
    main()
