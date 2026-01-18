from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
def validate_compressors(dummy: Any, value: Union[str, Iterable[str]]) -> list[str]:
    try:
        compressors = value.split(',')
    except AttributeError:
        compressors = list(value)
    for compressor in compressors[:]:
        if compressor not in _SUPPORTED_COMPRESSORS:
            compressors.remove(compressor)
            warnings.warn(f'Unsupported compressor: {compressor}', stacklevel=2)
        elif compressor == 'snappy' and (not _HAVE_SNAPPY):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with snappy is not available. You must install the python-snappy module for snappy support.', stacklevel=2)
        elif compressor == 'zlib' and (not _HAVE_ZLIB):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with zlib is not available. The zlib module is not available.', stacklevel=2)
        elif compressor == 'zstd' and (not _HAVE_ZSTD):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with zstandard is not available. You must install the zstandard module for zstandard support.', stacklevel=2)
    return compressors