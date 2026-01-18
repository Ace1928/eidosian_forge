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
def write_pack_index_v2(f, entries: Iterable[PackIndexEntry], pack_checksum: bytes) -> bytes:
    """Write a new pack index file.

    Args:
      f: File-like object to write to
      entries: List of tuples with object name (sha), offset_in_pack, and
        crc32_checksum.
      pack_checksum: Checksum of the pack file.
    Returns: The SHA of the index file written
    """
    f = SHA1Writer(f)
    f.write(b'\xfftOc')
    f.write(struct.pack('>L', 2))
    fan_out_table: Dict[int, int] = defaultdict(lambda: 0)
    for name, offset, entry_checksum in entries:
        fan_out_table[ord(name[:1])] += 1
    largetable: List[int] = []
    for i in range(256):
        f.write(struct.pack(b'>L', fan_out_table[i]))
        fan_out_table[i + 1] += fan_out_table[i]
    for name, offset, entry_checksum in entries:
        f.write(name)
    for name, offset, entry_checksum in entries:
        f.write(struct.pack(b'>L', entry_checksum))
    for name, offset, entry_checksum in entries:
        if offset < 2 ** 31:
            f.write(struct.pack(b'>L', offset))
        else:
            f.write(struct.pack(b'>L', 2 ** 31 + len(largetable)))
            largetable.append(offset)
    for offset in largetable:
        f.write(struct.pack(b'>Q', offset))
    assert len(pack_checksum) == 20
    f.write(pack_checksum)
    return f.write_sha()