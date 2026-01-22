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
class BaseTestPackIndexWriting:

    def assertSucceeds(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except ChecksumMismatch as e:
            self.fail(e)

    def index(self, filename, entries, pack_checksum):
        raise NotImplementedError(self.index)

    def test_empty(self):
        idx = self.index('empty.idx', [], pack_checksum)
        self.assertEqual(idx.get_pack_checksum(), pack_checksum)
        self.assertEqual(0, len(idx))

    def test_large(self):
        entry1_sha = hex_to_sha('4e6388232ec39792661e2e75db8fb117fc869ce6')
        entry2_sha = hex_to_sha('e98f071751bd77f59967bfa671cd2caebdccc9a2')
        entries = [(entry1_sha, 17480489991855577991, 24), (entry2_sha, ~17480489991855577991 & 2 ** 64 - 1, 92)]
        if not self._supports_large:
            self.assertRaises(TypeError, self.index, 'single.idx', entries, pack_checksum)
            return
        idx = self.index('single.idx', entries, pack_checksum)
        self.assertEqual(idx.get_pack_checksum(), pack_checksum)
        self.assertEqual(2, len(idx))
        actual_entries = list(idx.iterentries())
        self.assertEqual(len(entries), len(actual_entries))
        for mine, actual in zip(entries, actual_entries):
            my_sha, my_offset, my_crc = mine
            actual_sha, actual_offset, actual_crc = actual
            self.assertEqual(my_sha, actual_sha)
            self.assertEqual(my_offset, actual_offset)
            if self._has_crc32_checksum:
                self.assertEqual(my_crc, actual_crc)
            else:
                self.assertIsNone(actual_crc)

    def test_single(self):
        entry_sha = hex_to_sha('6f670c0fb53f9463760b7295fbb814e965fb20c8')
        my_entries = [(entry_sha, 178, 42)]
        idx = self.index('single.idx', my_entries, pack_checksum)
        self.assertEqual(idx.get_pack_checksum(), pack_checksum)
        self.assertEqual(1, len(idx))
        actual_entries = list(idx.iterentries())
        self.assertEqual(len(my_entries), len(actual_entries))
        for mine, actual in zip(my_entries, actual_entries):
            my_sha, my_offset, my_crc = mine
            actual_sha, actual_offset, actual_crc = actual
            self.assertEqual(my_sha, actual_sha)
            self.assertEqual(my_offset, actual_offset)
            if self._has_crc32_checksum:
                self.assertEqual(my_crc, actual_crc)
            else:
                self.assertIsNone(actual_crc)