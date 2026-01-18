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
def test_resize_images_parallel(self):
    try:
        asyncio.run(self.async_test_resize_images_parallel())
        LoggingManager.debug('test_resize_images_parallel executed successfully.')
    except Exception as e:
        LoggingManager.error(f'Error executing test_resize_images_parallel: {traceback.format_exc()}')