import sys
from pathlib import Path
import zstandard as zstd
from PIL import Image


def compress_image(input_path: Path, output_path: Path) -> None:
    """
    Compresses an image file using Zstandard compression.

    Args:
        input_path (Path): The path to the input image file.
        output_path (Path): The path where the compressed file will be saved.
    """
    # Open the image to ensure it's a valid image file
    with Image.open(input_path) as img:
        img_format = img.format.lower()

    # Read the original image data
    with open(input_path, "rb") as file:
        data = file.read()

    # Compress the data
    cctx = zstd.ZstdCompressor()
    compressed_data = cctx.compress(data)

    # Save the compressed data with a .format.zstd extension
    with open(output_path.with_suffix(f".{img_format}.zstd"), "wb") as file:
        file.write(compressed_data)

    print(f"Compressed file saved to: {output_path.with_suffix(f'.{img_format}.zstd')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compress_image.py <path_to_image>")
        sys.exit(1)

    input_image_path = Path(sys.argv[1])
    if not input_image_path.exists():
        print(f"Error: The file {input_image_path} does not exist.")
        sys.exit(1)

    output_image_path = input_image_path.with_suffix(input_image_path.suffix + ".zstd")
    compress_image(input_image_path, output_image_path)
