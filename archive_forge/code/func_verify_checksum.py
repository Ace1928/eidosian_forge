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
def verify_checksum(image_data: bytes, expected_checksum: str) -> bool:
    """
    Verifies the integrity of the image data against the expected checksum.

    This function computes the checksum of the provided image data and compares it to an expected value. The comparison
    result indicates whether the image data's integrity is intact.

    Args:
        image_data (bytes): The raw image data.
        expected_checksum (str): The expected checksum for verification.

    Returns:
        bool: True if the checksum matches, False otherwise.

    Raises:
        ImageOperationError: If checksum verification fails.
    """
    actual_checksum = get_image_hash(image_data)
    is_valid = actual_checksum == expected_checksum
    LoggingManager.debug(f'Checksum verification result: {is_valid}.')
    if not is_valid:
        raise ImageOperationError(f'Checksum verification failed. Expected {expected_checksum}, got {actual_checksum}.')
    return is_valid