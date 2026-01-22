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
class TestThinPack(PackTests):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.blobs = {}
        for blob in (b'foo', b'bar', b'foo1234', b'bar2468'):
            self.blobs[blob] = make_object(Blob, data=blob)
        self.store.add_object(self.blobs[b'foo'])
        self.store.add_object(self.blobs[b'bar'])
        self.pack_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.pack_dir)
        self.pack_prefix = os.path.join(self.pack_dir, 'pack')
        with open(self.pack_prefix + '.pack', 'wb') as f:
            build_pack(f, [(REF_DELTA, (self.blobs[b'foo'].id, b'foo1234')), (Blob.type_num, b'bar'), (REF_DELTA, (self.blobs[b'bar'].id, b'bar2468'))], store=self.store)
        with self.make_pack(True) as pack:
            with PackData(pack._data_path) as data:
                data.create_index(self.pack_prefix + '.idx', resolve_ext_ref=pack.resolve_ext_ref)
        del self.store[self.blobs[b'bar'].id]

    def make_pack(self, resolve_ext_ref):
        return Pack(self.pack_prefix, resolve_ext_ref=self.store.get_raw if resolve_ext_ref else None)

    def test_get_raw(self):
        with self.make_pack(False) as p:
            self.assertRaises(KeyError, p.get_raw, self.blobs[b'foo1234'].id)
        with self.make_pack(True) as p:
            self.assertEqual((3, b'foo1234'), p.get_raw(self.blobs[b'foo1234'].id))

    def test_get_unpacked_object(self):
        self.maxDiff = None
        with self.make_pack(False) as p:
            expected = UnpackedObject(7, delta_base=b'\x19\x10(\x15f=#\xf8\xb7ZG\xe7\xa0\x19e\xdc\xdc\x96F\x8c', decomp_chunks=[b'\x03\x07\x90\x03\x041234'])
            expected.offset = 12
            got = p.get_unpacked_object(self.blobs[b'foo1234'].id)
            self.assertEqual(expected, got)
        with self.make_pack(True) as p:
            expected = UnpackedObject(7, delta_base=b'\x19\x10(\x15f=#\xf8\xb7ZG\xe7\xa0\x19e\xdc\xdc\x96F\x8c', decomp_chunks=[b'\x03\x07\x90\x03\x041234'])
            expected.offset = 12
            got = p.get_unpacked_object(self.blobs[b'foo1234'].id)
            self.assertEqual(expected, got)

    def test_iterobjects(self):
        with self.make_pack(False) as p:
            self.assertRaises(UnresolvedDeltas, list, p.iterobjects())
        with self.make_pack(True) as p:
            self.assertEqual(sorted([self.blobs[b'foo1234'].id, self.blobs[b'bar'].id, self.blobs[b'bar2468'].id]), sorted((o.id for o in p.iterobjects())))