import unittest
import io
import os
import asyncio
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from typing import Tuple, Dict, Any, Optional, List, Callable
from core_services import (
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
import traceback
@log_function_call
def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    This function checks if the provided image meets certain criteria regarding its format and dimensions. It supports
    multiple image formats and ensures that the image's width and height do not exceed predefined limits.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.

    Raises:
        ImageOperationError: If image validation fails.
    """
    try:
        valid_formats = ['JPEG', 'PNG', 'BMP', 'GIF', 'TIFF']
        is_valid = image.format in valid_formats and image.width <= 4000 and (image.height <= 4000)
        LoggingManager.debug(f'Image validation result: {is_valid}.')
        return is_valid
    except Exception as e:
        LoggingManager.error(f'Error validating image: {e}')
        raise ImageOperationError(f'Error validating image due to: {str(e)}') from e