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
def _offset_v2(self, i):
    """:return: 32 or 64 byte offset into pack files. 64 byte offsets will only
            be returned if the pack is larger than 4 GiB, or 2^32"""
    offset = unpack_from('>L', self._cursor.map(), self._pack_offset + i * 4)[0]
    if offset & 2147483648:
        offset = unpack_from('>Q', self._cursor.map(), self._pack_64_offset + (offset & ~2147483648) * 8)[0]
    return offset