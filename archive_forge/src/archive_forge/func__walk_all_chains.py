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
def _walk_all_chains(self):
    for offset, type_num in self._full_ofs:
        yield from self._follow_chain(offset, type_num, None)
    yield from self._walk_ref_chains()
    assert not self._pending_ofs, repr(self._pending_ofs)