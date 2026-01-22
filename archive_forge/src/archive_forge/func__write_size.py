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
def _write_size(self, token, size):
    if size < 15:
        self._fp.write(struct.pack('>B', token | size))
    elif size < 1 << 8:
        self._fp.write(struct.pack('>BBB', token | 15, 16, size))
    elif size < 1 << 16:
        self._fp.write(struct.pack('>BBH', token | 15, 17, size))
    elif size < 1 << 32:
        self._fp.write(struct.pack('>BBL', token | 15, 18, size))
    else:
        self._fp.write(struct.pack('>BBQ', token | 15, 19, size))