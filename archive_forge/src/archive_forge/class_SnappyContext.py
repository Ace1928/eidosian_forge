from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
class SnappyContext:
    compressor_id = 1

    @staticmethod
    def compress(data: bytes) -> bytes:
        return snappy.compress(data)