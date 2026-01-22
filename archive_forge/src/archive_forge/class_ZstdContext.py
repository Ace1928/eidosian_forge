from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
class ZstdContext:
    compressor_id = 3

    @staticmethod
    def compress(data: bytes) -> bytes:
        return ZstdCompressor().compress(data)