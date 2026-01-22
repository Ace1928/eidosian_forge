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
def _follow_chain(self, offset: int, obj_type_num: int, base_chunks: List[bytes]):
    todo = [(offset, obj_type_num, base_chunks)]
    while todo:
        offset, obj_type_num, base_chunks = todo.pop()
        unpacked = self._resolve_object(offset, obj_type_num, base_chunks)
        yield self._result(unpacked)
        unblocked = chain(self._pending_ofs.pop(unpacked.offset, []), self._pending_ref.pop(unpacked.sha(), []))
        todo.extend(((new_offset, unpacked.obj_type_num, unpacked.obj_chunks) for new_offset in unblocked))