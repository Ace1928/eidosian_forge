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
class EncodeCopyOperationTests(TestCase):

    def test_basic(self):
        self.assertEqual(b'\x80', _encode_copy_operation(0, 0))
        self.assertEqual(b'\x91\x01\n', _encode_copy_operation(1, 10))
        self.assertEqual(b'\xb1d\xe8\x03', _encode_copy_operation(100, 1000))
        self.assertEqual(b'\x93\xe8\x03\x01', _encode_copy_operation(1000, 1))