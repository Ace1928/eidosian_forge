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
def test_write_pack_header(self):
    f = BytesIO()
    write_pack_header(f.write, 42)
    self.assertEqual(b'PACK\x00\x00\x00\x02\x00\x00\x00*', f.getvalue())