from __future__ import annotations
import zlib
from .base import BaseCompression, logger
from typing import Optional
def validate_compression_level(self):
    """
        Validates the compression level
        """
    assert self.compression_level == -1 or self.compression_level in range(10), 'Compression level must be between 0 and 9 or -1'