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
def _entry_v1(self, i):
    """:return: tuple(offset, binsha, 0)"""
    return unpack_from('>L20s', self._cursor.map(), 1024 + i * 24) + (0,)