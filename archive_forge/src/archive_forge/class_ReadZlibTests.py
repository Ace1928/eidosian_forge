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
class ReadZlibTests(TestCase):
    decomp = b"tree 4ada885c9196b6b6fa08744b5862bf92896fc002\nparent None\nauthor Jelmer Vernooij <jelmer@samba.org> 1228980214 +0000\ncommitter Jelmer Vernooij <jelmer@samba.org> 1228980214 +0000\n\nProvide replacement for mmap()'s offset argument."
    comp = zlib.compress(decomp)
    extra = b'nextobject'

    def setUp(self):
        super().setUp()
        self.read = BytesIO(self.comp + self.extra).read
        self.unpacked = UnpackedObject(Tree.type_num, decomp_len=len(self.decomp), crc32=0)

    def test_decompress_size(self):
        good_decomp_len = len(self.decomp)
        self.unpacked.decomp_len = -1
        self.assertRaises(ValueError, read_zlib_chunks, self.read, self.unpacked)
        self.unpacked.decomp_len = good_decomp_len - 1
        self.assertRaises(zlib.error, read_zlib_chunks, self.read, self.unpacked)
        self.unpacked.decomp_len = good_decomp_len + 1
        self.assertRaises(zlib.error, read_zlib_chunks, self.read, self.unpacked)

    def test_decompress_truncated(self):
        read = BytesIO(self.comp[:10]).read
        self.assertRaises(zlib.error, read_zlib_chunks, read, self.unpacked)
        read = BytesIO(self.comp).read
        self.assertRaises(zlib.error, read_zlib_chunks, read, self.unpacked)

    def test_decompress_empty(self):
        unpacked = UnpackedObject(Tree.type_num, decomp_len=0)
        comp = zlib.compress(b'')
        read = BytesIO(comp + self.extra).read
        unused = read_zlib_chunks(read, unpacked)
        self.assertEqual(b'', b''.join(unpacked.decomp_chunks))
        self.assertNotEqual(b'', unused)
        self.assertEqual(self.extra, unused + read())

    def test_decompress_no_crc32(self):
        self.unpacked.crc32 = None
        read_zlib_chunks(self.read, self.unpacked)
        self.assertEqual(None, self.unpacked.crc32)

    def _do_decompress_test(self, buffer_size, **kwargs):
        unused = read_zlib_chunks(self.read, self.unpacked, buffer_size=buffer_size, **kwargs)
        self.assertEqual(self.decomp, b''.join(self.unpacked.decomp_chunks))
        self.assertEqual(zlib.crc32(self.comp), self.unpacked.crc32)
        self.assertNotEqual(b'', unused)
        self.assertEqual(self.extra, unused + self.read())

    def test_simple_decompress(self):
        self._do_decompress_test(4096)
        self.assertEqual(None, self.unpacked.comp_chunks)

    def test_decompress_buffer_size_1(self):
        self._do_decompress_test(1)

    def test_decompress_buffer_size_2(self):
        self._do_decompress_test(2)

    def test_decompress_buffer_size_3(self):
        self._do_decompress_test(3)

    def test_decompress_buffer_size_4(self):
        self._do_decompress_test(4)

    def test_decompress_include_comp(self):
        self._do_decompress_test(4096, include_comp=True)
        self.assertEqual(self.comp, b''.join(self.unpacked.comp_chunks))