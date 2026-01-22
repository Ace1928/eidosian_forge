import array
import contextlib
import enum
import struct
@property
def LastValue(self):
    return self._stack[-1]