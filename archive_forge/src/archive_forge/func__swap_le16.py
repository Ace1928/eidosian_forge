import os as _os
import sys as _sys
import warnings as _warnings
from .base import Sign
from .controller_db import mapping_list
def _swap_le16(value):
    """Ensure 16bit value is in Big Endian format"""
    if _sys.byteorder == 'little':
        return (value << 8 | value >> 8) & 65535
    return value