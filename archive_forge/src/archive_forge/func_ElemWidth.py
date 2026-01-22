import array
import contextlib
import enum
import struct
def ElemWidth(self, buf_size, elem_index=0):
    if Type.IsInline(self._type):
        return self._min_bit_width
    for byte_width in (1, 2, 4, 8):
        offset_loc = buf_size + _PaddingBytes(buf_size, byte_width) + elem_index * byte_width
        bit_width = BitWidth.U(offset_loc - self._value)
        if byte_width == 1 << bit_width:
            return bit_width
    raise ValueError('relative offset is too big')