from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class SizeLimitError(ValueError):
    """Raised when trying to (de-)serialise data exceeding D-Bus' size limit.

    This is currently only implemented for arrays, where the maximum size is
    64 MiB.
    """
    pass