from datetime import datetime as _DateTime
import sys
import struct
from .exceptions import BufferFull, OutOfData, ExtraData, FormatError, StackError
from .ext import ExtType, Timestamp
def _pack_array_header(self, n):
    if n <= 15:
        return self._buffer.write(struct.pack('B', 144 + n))
    if n <= 65535:
        return self._buffer.write(struct.pack('>BH', 220, n))
    if n <= 4294967295:
        return self._buffer.write(struct.pack('>BI', 221, n))
    raise ValueError('Array is too large')