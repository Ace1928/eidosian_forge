import array
import contextlib
import enum
import struct
def MutateString(self, value):
    return String(self._Indirect(), self._byte_width).Mutate(value)