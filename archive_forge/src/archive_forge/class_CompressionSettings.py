from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
class CompressionSettings:

    def __init__(self, compressors: list[str], zlib_compression_level: int):
        self.compressors = compressors
        self.zlib_compression_level = zlib_compression_level

    def get_compression_context(self, compressors: Optional[list[str]]) -> Union[SnappyContext, ZlibContext, ZstdContext, None]:
        if compressors:
            chosen = compressors[0]
            if chosen == 'snappy':
                return SnappyContext()
            elif chosen == 'zlib':
                return ZlibContext(self.zlib_compression_level)
            elif chosen == 'zstd':
                return ZstdContext()
            return None
        return None