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
def test_object_sha1(self):
    """Tests that the correct object offset is returned from the index."""
    p = self.get_pack_index(pack1_sha)
    self.assertRaises(KeyError, p.object_sha1, 876)
    self.assertEqual(p.object_sha1(178), hex_to_sha(a_sha))
    self.assertEqual(p.object_sha1(138), hex_to_sha(tree_sha))
    self.assertEqual(p.object_sha1(12), hex_to_sha(commit_sha))