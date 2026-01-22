import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class BytesIO_EOF(object):
    """This class differs from BytesIO in that a ValueError exception is
    raised whenever EOF is reached."""

    def __init__(self, initial_bytes):
        self._buffer = initial_bytes
        self._index = 0
        self._bookmark = None

    def set_bookmark(self):
        self._bookmark = self._index

    def data_since_bookmark(self):
        assert self._bookmark is not None
        return self._buffer[self._bookmark:self._index]

    def remaining_data(self):
        return len(self._buffer) - self._index

    def read(self, length):
        new_index = self._index + length
        if new_index > len(self._buffer):
            raise ValueError('Not enough data for DER decoding: expected %d bytes and found %d' % (new_index, len(self._buffer)))
        result = self._buffer[self._index:new_index]
        self._index = new_index
        return result

    def read_byte(self):
        return bord(self.read(1)[0])