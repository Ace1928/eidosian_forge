import array
import contextlib
import enum
import struct
class BitWidth(enum.IntEnum):
    """Supported bit widths of value types.

  These are used in the lower 2 bits of a type field to determine the size of
  the elements (and or size field) of the item pointed to (e.g. vector).
  """
    W8 = 0
    W16 = 1
    W32 = 2
    W64 = 3

    @staticmethod
    def U(value):
        """Returns the minimum `BitWidth` to encode unsigned integer value."""
        assert value >= 0
        if value < 1 << 8:
            return BitWidth.W8
        elif value < 1 << 16:
            return BitWidth.W16
        elif value < 1 << 32:
            return BitWidth.W32
        elif value < 1 << 64:
            return BitWidth.W64
        else:
            raise ValueError('value is too big to encode: %s' % value)

    @staticmethod
    def I(value):
        """Returns the minimum `BitWidth` to encode signed integer value."""
        value *= 2
        return BitWidth.U(value if value >= 0 else ~value)

    @staticmethod
    def F(value):
        """Returns the `BitWidth` to encode floating point value."""
        if struct.unpack('<f', struct.pack('<f', value))[0] == value:
            return BitWidth.W32
        return BitWidth.W64

    @staticmethod
    def B(byte_width):
        return {1: BitWidth.W8, 2: BitWidth.W16, 4: BitWidth.W32, 8: BitWidth.W64}[byte_width]