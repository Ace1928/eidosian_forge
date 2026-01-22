import array
import contextlib
import enum
import struct
@property
def IsBool(self):
    return self._type is Type.BOOL