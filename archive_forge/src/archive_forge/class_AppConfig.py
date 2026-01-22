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
class AppConfig:
    """
    A configuration class that holds application-wide constants and settings for image processing.
    """
    MAX_SIZE: Tuple[int, int] = (800, 600)
    ENHANCEMENT_FACTOR: float = 1.5