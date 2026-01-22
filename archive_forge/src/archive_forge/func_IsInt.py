import array
import contextlib
import enum
import struct
@property
def IsInt(self):
    return self._type in (Type.INT, Type.INDIRECT_INT, Type.UINT, Type.INDIRECT_UINT)