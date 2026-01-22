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
def _sha_to_index(self, sha):
    """:return: index for the given sha, or raise"""
    index = self._index.sha_to_index(sha)
    if index is None:
        raise BadObject(sha)
    return index