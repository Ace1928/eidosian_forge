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
def _do_decompress_test(self, buffer_size, **kwargs):
    unused = read_zlib_chunks(self.read, self.unpacked, buffer_size=buffer_size, **kwargs)
    self.assertEqual(self.decomp, b''.join(self.unpacked.decomp_chunks))
    self.assertEqual(zlib.crc32(self.comp), self.unpacked.crc32)
    self.assertNotEqual(b'', unused)
    self.assertEqual(self.extra, unused + self.read())