import array
import contextlib
import enum
import struct
@InMap
def ReuseValue(self, value):
    self._stack.append(value)