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
def test_resize_image(self) -> None:
    """
        Confirms the image resizing functionality.

        This test verifies that the resize_image function correctly resizes the test image to the specified dimensions,
        ensuring that the resized image's width and height do not exceed the maximum allowed dimensions. The test asserts
        that the resized image is an instance of Image.Image and that its dimensions are within the specified limits.
        """
    image: Image.Image = ensure_image_format(self.test_image_data)
    resized_image: Image.Image = resize_image(image)
    self.assertIsInstance(resized_image, Image.Image)
    self.assertTrue(resized_image.width <= 800 and resized_image.height <= 600)