import array
import contextlib
import enum
import struct
def Slice(self, offset):
    """Returns new `Buf` which starts from the given offset."""
    return Buf(self._buf, self._offset + offset)