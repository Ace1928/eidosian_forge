from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _unpack_from(f, b, o=0):
    """Explicit type cast for legacy struct.unpack_from"""
    return struct.unpack_from(f, bytes(b), o)