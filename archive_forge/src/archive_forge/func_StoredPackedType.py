import array
import contextlib
import enum
import struct
def StoredPackedType(self, parent_bit_width=BitWidth.W8):
    return Type.Pack(self._type, self.StoredWidth(parent_bit_width))