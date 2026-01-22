import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
def _read_ints(self, n, size):
    data = self._fp.read(size * n)
    if size in _BINARY_FORMAT:
        return struct.unpack(f'>{n}{_BINARY_FORMAT[size]}', data)
    else:
        if not size or len(data) != size * n:
            raise InvalidFileException()
        return tuple((int.from_bytes(data[i:i + size], 'big') for i in range(0, size * n, size)))