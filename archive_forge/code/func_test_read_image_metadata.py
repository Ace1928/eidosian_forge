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
def test_read_image_metadata(self):
    metadata = read_image_metadata(self.test_image_data)
    self.assertIsInstance(metadata, dict)