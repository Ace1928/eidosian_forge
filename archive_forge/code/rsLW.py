# image_processing.py
import configparser
import os
from PIL import Image, ImageEnhance, ExifTags
import io
import zstd  # Updated to use zstd for compression
import hashlib
from cryptography.fernet import Fernet
from typing import Optional
import logging
from logging_config import configure_logging


# Ensure config and read values
def ensure_config(section, default_values):
    config_path = "config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section(section):
        config.add_section(section)
        for key, value in default_values.items():
            config.set(section, key, value)
        with open(config_path, "w") as configfile:
            config.write(configfile)
    return config


config = ensure_config(
    "ImageProcessing", {"MaxSize": "800,600", "EnhancementFactor": "1.5"}
)
MAX_SIZE = tuple(map(int, config.get("ImageProcessing", "MaxSize").split(",")))
ENHANCEMENT_FACTOR = float(config.get("ImageProcessing", "EnhancementFactor"))

configure_logging()


class BaseImageOperation:
    """
    Base class for image operations.
    """

    def process(self, image_data: bytes) -> bytes:
        """
        Process image data. This method should be overridden by subclasses.

        Args:
            image_data (bytes): The raw image data.

        Returns:
            bytes: The processed image data.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ContrastEnhancement(BaseImageOperation):
    """
    Enhances the contrast of an image.
    """

    def process(
        self, image_data: bytes, enhancement_factor: float = ENHANCEMENT_FACTOR
    ) -> bytes:
        try:
            image = Image.open(io.BytesIO(image_data))
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(enhancement_factor)
            with io.BytesIO() as output:
                enhanced_image.save(output, format=image.format)
                return output.getvalue()
        except Exception as e:
            logging.error(f"Error enhancing image contrast: {e}")
            raise


def resize_image(image: Image.Image, max_size: tuple = MAX_SIZE) -> Image.Image:
    """
    Resizes an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The original image.
        max_size (tuple): A tuple of (max_width, max_height).

    Returns:
        Image.Image: The resized image.
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logging.debug(f"Image resized to max size {max_size}.")
        return image
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        raise


def get_image_hash(image_data: bytes) -> str:
    """
    Hashes the image data using SHA-512 to ensure uniqueness and increased security.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal hash of the image.
    """
    try:
        sha512_hash = hashlib.sha512()
        sha512_hash.update(image_data)
        logging.debug("Image hash successfully generated using SHA-512.")
        return sha512_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error generating image hash with SHA-512: {e}")
        raise


def compress_and_encrypt_image_data(
    image_data: bytes, image_format: str, cipher_suite: Fernet
) -> bytes:
    """
    Compresses and encrypts image data, prepending the image format before compression.

    Args:
        image_data (bytes): The raw image data.
        image_format (str): The format of the image (e.g., 'JPEG').
        cipher_suite (Fernet): The cipher suite for encryption.

    Returns:
        bytes: The compressed and encrypted image data.
    """
    try:
        formatted_data = image_format.encode() + b"\x00" + image_data
        compressed_data = zstd.compress(formatted_data)
        encrypted_data = cipher_suite.encrypt(compressed_data)
        logging.debug("Image data compressed with zstd and encrypted successfully.")
        return encrypted_data
    except Exception as e:
        logging.error(f"Error in compressing and encrypting image data: {e}")
        raise


def decrypt_and_decompress_image_data(
    encrypted_data: bytes, cipher_suite: Fernet
) -> tuple:
    """
    Decrypts and decompresses image data, returning both the image data and its format.

    Args:
        encrypted_data (bytes): The encrypted image data.
        cipher_suite (Fernet): The cipher suite for decryption.

    Returns:
        tuple: A tuple containing the decompressed image data and its format.
    """
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        decompressed_data = zstd.decompress(decrypted_data)
        image_format, image_data = decompressed_data.split(b"\x00", 1)
        logging.debug("Image data decrypted and decompressed with zstd successfully.")
        return image_data, image_format.decode()
    except Exception as e:
        logging.error(f"Error in decrypting and decompressing image data: {e}")
        raise


def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        valid_formats = ["JPEG", "PNG", "BMP", "GIF"]  # Expanded list
        is_valid = (
            image.format in valid_formats
            and image.width <= 4000
            and image.height <= 4000
        )
        logging.debug(f"Image validation result: {is_valid}.")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        raise


def is_valid_image(image_data: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(image_data))
        return (
            image.format in ["JPEG", "PNG", "BMP", "GIF"]
            and image.width <= 4000
            and image.height <= 4000
        )
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        return False


def read_image_metadata(image_data: bytes) -> dict:
    """
    Reads EXIF metadata from an image.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        dict: A dictionary containing EXIF metadata, if available.
    """
    with Image.open(io.BytesIO(image_data)) as image:
        exif_data = {}
        if hasattr(image, "_getexif"):
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
        return exif_data
