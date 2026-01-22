import array
import contextlib
import enum
import struct
@InMap
def IndirectFloat(self, value, byte_width=0):
    """Encodes floating point value indirectly.

    Args:
      value: A floating point value.
      byte_width: Number of bytes to use: 4 or 8.
    """
    bit_width = BitWidth.F(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._PushIndirect(value, Type.INDIRECT_FLOAT, bit_width)