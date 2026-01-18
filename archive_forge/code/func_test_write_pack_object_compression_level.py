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
def test_write_pack_object_compression_level(self):
    f = BytesIO()
    f.write(b'header')
    offset = f.tell()
    sha_a = sha1(b'foo')
    sha_b = sha_a.copy()
    write_pack_object(f.write, Blob.type_num, b'blob', sha=sha_a, compression_level=6)
    self.assertNotEqual(sha_a.digest(), sha_b.digest())
    sha_b.update(f.getvalue()[offset:])
    self.assertEqual(sha_a.digest(), sha_b.digest())