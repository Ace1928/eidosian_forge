"""
Image Processing Module

This module provides functionalities for processing images, including format verification, contrast enhancement, resizing, hashing, compression, encryption, decryption, and decompression. It serves as a core component of the image interconversion GUI application, facilitating secure and efficient image manipulation.

Author: Lloyd Handyside
Creation Date: 2024-04-08
Last Modified: 2024-04-08

Functionalities:
- Ensure image format compatibility
- Enhance image contrast
- Resize images while maintaining aspect ratio
- Generate image hashes
- Compress and decompress image data
- Encrypt and decrypt image data
- Validate image properties
- Read image metadata
"""

import io
import os
import logging
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from cryptography.fernet import Fernet
from typing import Tuple, Dict, Any, Optional
from config_manager import ConfigManager  # Assuming proper path
from logging_config import configure_logging

# Initialize logging
configure_logging()

# Load configurations using ConfigManager
config_manager = ConfigManager()
config_path = os.path.join(os.path.dirname(__file__), "config.ini")
config_manager.load_config(config_path, "ImageProcessing")
MAX_SIZE: Tuple[int, int] = tuple(
    map(
        int,
        config_manager.get(
            "main_config", "ImageProcessing", "MaxSize", fallback="800,600"
        ).split(","),
    )
)
ENHANCEMENT_FACTOR: float = float(
    config_manager.get(
        "main_config", "ImageProcessing", "EnhancementFactor", fallback="1.5"
    )
)


class ImageOperationError(Exception):
    """Custom exception for image operation errors."""


def log_function_call(func):
    """
    A decorator that logs the entry and exit of functions.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with logging.
    """

    def wrapper(*args, **kwargs):
        logging.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logging.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


@log_function_call
def ensure_image_format(image_data: bytes) -> Image.Image:
    """
    Ensures the given image data can be opened and returns the Image object.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Image.Image: The opened image.

    Raises:
        ImageOperationError: If the image cannot be opened.
    """
    try:
        image: Image.Image = Image.open(io.BytesIO(image_data))
        logging.debug("Image format ensured successfully.")
        return image
    except Exception as e:
        logging.error(f"Failed to open image: {e}")
        raise ImageOperationError(f"Failed to open image: {e}")


@log_function_call
def enhance_contrast(
    image: Image.Image, enhancement_factor: float = ENHANCEMENT_FACTOR
) -> bytes:
    """
    Enhances the contrast of an image.

    Args:
        image (Image.Image): The image to enhance.
        enhancement_factor (float, optional): The factor by which to enhance the image's contrast. Defaults to ENHANCEMENT_FACTOR.

    Returns:
        bytes: The enhanced image data.

    Raises:
        ImageOperationError: If contrast enhancement fails.
    """
    try:
        enhancer: ImageEnhance.Contrast = ImageEnhance.Contrast(image)
        enhanced_image: Image.Image = enhancer.enhance(enhancement_factor)
        with io.BytesIO() as output:
            enhanced_image.save(output, format=image.format)
            logging.debug("Image contrast enhanced successfully.")
            return output.getvalue()
    except Exception as e:
        logging.error(f"Error enhancing image contrast: {e}")
        raise ImageOperationError(f"Error enhancing image contrast: {e}")


@log_function_call
def resize_image(
    image: Image.Image, max_size: Tuple[int, int] = MAX_SIZE
) -> Image.Image:
    """
    Resizes an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The original image.
        max_size (Tuple[int, int], optional): A tuple of (max_width, max_height). Defaults to MAX_SIZE.

    Returns:
        Image.Image: The resized image.

    Raises:
        ImageOperationError: If resizing the image fails.
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logging.debug(f"Image resized to max size {max_size}.")
        return image
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        raise ImageOperationError(f"Error resizing image: {e}")


@log_function_call
def get_image_hash(image_data: bytes) -> str:
    """
    Generates a SHA-512 hash of the image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal hash of the image.
    """
    sha512_hash = hashlib.sha512()
    sha512_hash.update(image_data)
    logging.debug("Image hash successfully generated using SHA-512.")
    return sha512_hash.hexdigest()


@log_function_call
def compress(image_data: bytes, image_format: str) -> bytes:
    """
    Compresses image data along with its format and a checksum for integrity verification.

    Args:
        image_data (bytes): The raw image data.
        image_format (str): The format of the image.

    Returns:
        bytes: The compressed image data.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    formatted_data = f"{image_format}\x00{checksum}\x00".encode() + image_data
    compressed_data = zstd.compress(formatted_data)
    logging.debug("Image data compressed successfully.")
    return compressed_data


@log_function_call
def encrypt(data: bytes, cipher_suite: Fernet) -> bytes:
    """
    Encrypts data using the provided cipher suite.

    Args:
        data (bytes): The data to encrypt.
        cipher_suite (Fernet): The cipher suite for encryption.

    Returns:
        bytes: The encrypted data.
    """
    encrypted_data = cipher_suite.encrypt(data)
    logging.debug("Data encrypted successfully.")
    return encrypted_data


@log_function_call
def decrypt(encrypted_data: bytes, cipher_suite: Fernet) -> bytes:
    """
    Decrypts data using the provided cipher suite.

    Args:
        encrypted_data (bytes): The encrypted data.
        cipher_suite (Fernet): The cipher suite for decryption.

    Returns:
        bytes: The decrypted data.
    """
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    logging.debug("Data decrypted successfully.")
    return decrypted_data


@log_function_call
def decompress(data: bytes) -> Tuple[bytes, str]:
    """
    Decompresses data and extracts the image format and raw image data.

    Args:
        data (bytes): The compressed data.

    Returns:
        Tuple[bytes, str]: The raw image data and its format.
    """
    decompressed_data = zstd.decompress(data)
    image_format, checksum, image_data = decompressed_data.split(b"\x00", 2)
    logging.debug("Data decompressed successfully.")
    return image_data, image_format.decode()


@log_function_call
def generate_checksum(image_data: bytes) -> str:
    """
    Generates a SHA-256 checksum for the given image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal checksum of the image.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    logging.debug("Checksum generated successfully.")
    return checksum


@log_function_call
def verify_checksum(image_data: bytes, expected_checksum: str) -> bool:
    """
    Verifies the integrity of the image data against the expected checksum.

    Args:
        image_data (bytes): The raw image data.
        expected_checksum (str): The expected checksum for verification.

    Returns:
        bool: True if the checksum matches, False otherwise.
    """
    actual_checksum = hashlib.sha256(image_data).hexdigest()
    is_valid = actual_checksum == expected_checksum
    logging.debug(f"Checksum verification result: {is_valid}.")
    return is_valid


@log_function_call
def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.

    Raises:
        ImageOperationError: If image validation fails.
    """
    try:
        valid_formats = ["JPEG", "PNG", "BMP", "GIF"]
        is_valid = (
            image.format in valid_formats
            and image.width <= 4000
            and image.height <= 4000
        )
        logging.debug(f"Image validation result: {is_valid}.")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        raise ImageOperationError(f"Error validating image: {e}")


@log_function_call
def read_image_metadata(image_data: bytes) -> Dict[str, Any]:
    """
    Reads EXIF metadata from an image.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Dict[str, Any]: A dictionary containing EXIF metadata, if available.

    Raises:
        ImageOperationError: If reading metadata fails.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            exif_data = {}
            if hasattr(image, "_getexif"):
                exif = image._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = value
            logging.debug("Image metadata read successfully.")
            return exif_data
    except Exception as e:
        logging.error(f"Error reading image metadata: {e}")
        raise ImageOperationError(f"Error reading image metadata: {e}")


# TODO:
# - Implement support for additional image formats.
# - Optimize performance for large image files.
# - Enhance encryption mechanisms for increased security.

# Known Issues:
# - Compression may result in quality loss for certain image formats.
