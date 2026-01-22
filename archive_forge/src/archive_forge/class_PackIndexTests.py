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
class PackIndexTests(PackTests):
    """Class that tests the index of packfiles."""

    def test_object_offset(self):
        """Tests that the correct object offset is returned from the index."""
        p = self.get_pack_index(pack1_sha)
        self.assertRaises(KeyError, p.object_offset, pack1_sha)
        self.assertEqual(p.object_offset(a_sha), 178)
        self.assertEqual(p.object_offset(tree_sha), 138)
        self.assertEqual(p.object_offset(commit_sha), 12)

    def test_object_sha1(self):
        """Tests that the correct object offset is returned from the index."""
        p = self.get_pack_index(pack1_sha)
        self.assertRaises(KeyError, p.object_sha1, 876)
        self.assertEqual(p.object_sha1(178), hex_to_sha(a_sha))
        self.assertEqual(p.object_sha1(138), hex_to_sha(tree_sha))
        self.assertEqual(p.object_sha1(12), hex_to_sha(commit_sha))

    def test_index_len(self):
        p = self.get_pack_index(pack1_sha)
        self.assertEqual(3, len(p))

    def test_get_stored_checksum(self):
        p = self.get_pack_index(pack1_sha)
        self.assertEqual(b'f2848e2ad16f329ae1c92e3b95e91888daa5bd01', sha_to_hex(p.get_stored_checksum()))
        self.assertEqual(b'721980e866af9a5f93ad674144e1459b8ba3e7b7', sha_to_hex(p.get_pack_checksum()))

    def test_index_check(self):
        p = self.get_pack_index(pack1_sha)
        self.assertSucceeds(p.check)

    def test_iterentries(self):
        p = self.get_pack_index(pack1_sha)
        entries = [(sha_to_hex(s), o, c) for s, o, c in p.iterentries()]
        self.assertEqual([(b'6f670c0fb53f9463760b7295fbb814e965fb20c8', 178, None), (b'b2a2766a2879c209ab1176e7e778b81ae422eeaa', 138, None), (b'f18faa16531ac570a3fdc8c7ca16682548dafd12', 12, None)], entries)

    def test_iter(self):
        p = self.get_pack_index(pack1_sha)
        self.assertEqual({tree_sha, commit_sha, a_sha}, set(p))