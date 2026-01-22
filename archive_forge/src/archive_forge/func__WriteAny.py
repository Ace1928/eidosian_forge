import array
import contextlib
import enum
import struct
def _WriteAny(self, value, byte_width):
    fmt = {Type.NULL: U, Type.BOOL: U, Type.INT: I, Type.UINT: U, Type.FLOAT: F}.get(value.Type)
    if fmt:
        self._Write(fmt, value.Value, byte_width)
    else:
        self._WriteOffset(value.Value, byte_width)