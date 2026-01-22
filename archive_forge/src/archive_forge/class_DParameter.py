from collections import namedtuple
from enum import IntEnum
from functools import lru_cache
from ._zstd import *
from . import _zstd
class DParameter(IntEnum):
    """Decompression parameters"""
    windowLogMax = _zstd._ZSTD_d_windowLogMax

    @lru_cache(maxsize=None)
    def bounds(self):
        """Return lower and upper bounds of a decompression parameter, both inclusive."""
        return _zstd._get_param_bounds(0, self.value)