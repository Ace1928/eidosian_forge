import array
import contextlib
import enum
import struct
def StoredWidth(self, parent_bit_width=BitWidth.W8):
    if Type.IsInline(self._type):
        return max(self._min_bit_width, parent_bit_width)
    return self._min_bit_width