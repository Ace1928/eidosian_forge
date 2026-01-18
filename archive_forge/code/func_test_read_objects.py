import os
import shutil
import sys
import tempfile
import zlib
from hashlib import sha1
from io import BytesIO
from typing import Set
from dulwich.tests import TestCase
from ..errors import ApplyDeltaError, ChecksumMismatch
from ..file import GitFile
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, hex_to_sha, sha_to_hex
from ..pack import (
from .utils import build_pack, make_object
def test_read_objects(self):
    f = BytesIO()
    entries = build_pack(f, [(Blob.type_num, b'blob'), (OFS_DELTA, (0, b'blob1'))])
    reader = PackStreamReader(f.read)
    objects = list(reader.read_objects(compute_crc32=True))
    self.assertEqual(2, len(objects))
    unpacked_blob, unpacked_delta = objects
    self.assertEqual(entries[0][0], unpacked_blob.offset)
    self.assertEqual(Blob.type_num, unpacked_blob.pack_type_num)
    self.assertEqual(Blob.type_num, unpacked_blob.obj_type_num)
    self.assertEqual(None, unpacked_blob.delta_base)
    self.assertEqual(b'blob', b''.join(unpacked_blob.decomp_chunks))
    self.assertEqual(entries[0][4], unpacked_blob.crc32)
    self.assertEqual(entries[1][0], unpacked_delta.offset)
    self.assertEqual(OFS_DELTA, unpacked_delta.pack_type_num)
    self.assertEqual(None, unpacked_delta.obj_type_num)
    self.assertEqual(unpacked_delta.offset - unpacked_blob.offset, unpacked_delta.delta_base)
    delta = create_delta(b'blob', b'blob1')
    self.assertEqual(b''.join(delta), b''.join(unpacked_delta.decomp_chunks))
    self.assertEqual(entries[1][4], unpacked_delta.crc32)