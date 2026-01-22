import array
import contextlib
import enum
import struct
class Buf:
    """Class to access underlying buffer object starting from the given offset."""

    def __init__(self, buf, offset):
        self._buf = buf
        self._offset = offset if offset >= 0 else len(buf) + offset
        self._length = len(buf) - self._offset

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._buf[_ShiftSlice(key, self._offset, self._length)]
        elif isinstance(key, int):
            return self._buf[self._offset + key]
        else:
            raise TypeError('invalid key type')

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._buf[_ShiftSlice(key, self._offset, self._length)] = value
        elif isinstance(key, int):
            self._buf[self._offset + key] = key
        else:
            raise TypeError('invalid key type')

    def __repr__(self):
        return 'buf[%d:]' % self._offset

    def Find(self, sub):
        """Returns the lowest index where the sub subsequence is found."""
        return self._buf[self._offset:].find(sub)

    def Slice(self, offset):
        """Returns new `Buf` which starts from the given offset."""
        return Buf(self._buf, self._offset + offset)

    def Indirect(self, offset, byte_width):
        """Return new `Buf` based on the encoded offset (indirect encoding)."""
        return self.Slice(offset - _Unpack(U, self[offset:offset + byte_width]))