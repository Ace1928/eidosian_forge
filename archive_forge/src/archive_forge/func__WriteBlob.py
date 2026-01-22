import array
import contextlib
import enum
import struct
def _WriteBlob(self, data, append_zero, type_):
    bit_width = BitWidth.U(len(data))
    byte_width = self._Align(bit_width)
    self._Write(U, len(data), byte_width)
    loc = len(self._buf)
    self._buf.extend(data)
    if append_zero:
        self._buf.append(0)
    self._stack.append(Value(loc, type_, bit_width))
    return loc