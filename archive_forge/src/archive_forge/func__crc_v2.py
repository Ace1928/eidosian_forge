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
def _crc_v2(self, i):
    """:return: 4 bytes crc for the object at index i"""
    return unpack_from('>L', self._cursor.map(), self._crc_list_offset + i * 4)[0]