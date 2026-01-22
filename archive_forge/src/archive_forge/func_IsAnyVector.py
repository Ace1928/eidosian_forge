import array
import contextlib
import enum
import struct
@property
def IsAnyVector(self):
    return self.IsVector or self.IsTypedVector or self.IsFixedTypedVector()