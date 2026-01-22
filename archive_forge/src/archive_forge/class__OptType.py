from __future__ import annotations
import errno
import sys
from enum import Enum, IntEnum, IntFlag
class _OptType(Enum):
    int = 'int'
    int64 = 'int64'
    bytes = 'bytes'
    fd = 'fd'