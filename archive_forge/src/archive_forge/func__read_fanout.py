import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
def _read_fanout(self, byte_offset):
    """Generate a fanout table from our data"""
    d = self._cursor.map()
    out = list()
    append = out.append
    for i in range(256):
        append(unpack_from('>L', d, byte_offset + i * 4)[0])
    return out