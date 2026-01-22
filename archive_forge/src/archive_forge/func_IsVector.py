import array
import contextlib
import enum
import struct
@property
def IsVector(self):
    return self._type in (Type.VECTOR, Type.MAP)