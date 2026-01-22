import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
def _get_size(self):
    if self._size is not None:
        return self._size
    self._size = os.path.getsize(self._filename)
    if self._size < self._header_size:
        errmsg = '%s is too small for a packfile (%d < %d)' % (self._filename, self._size, self._header_size)
        raise AssertionError(errmsg)
    return self._size