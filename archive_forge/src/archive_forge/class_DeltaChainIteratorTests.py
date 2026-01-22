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
class DeltaChainIteratorTests(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.fetched = set()

    def store_blobs(self, blobs_data):
        blobs = []
        for data in blobs_data:
            blob = make_object(Blob, data=data)
            blobs.append(blob)
            self.store.add_object(blob)
        return blobs

    def get_raw_no_repeat(self, bin_sha):
        """Wrapper around store.get_raw that doesn't allow repeat lookups."""
        hex_sha = sha_to_hex(bin_sha)
        self.assertNotIn(hex_sha, self.fetched, 'Attempted to re-fetch object %s' % hex_sha)
        self.fetched.add(hex_sha)
        return self.store.get_raw(hex_sha)

    def make_pack_iter(self, f, thin=None):
        if thin is None:
            thin = bool(list(self.store))
        resolve_ext_ref = thin and self.get_raw_no_repeat or None
        data = PackData('test.pack', file=f)
        return TestPackIterator.for_pack_data(data, resolve_ext_ref=resolve_ext_ref)

    def make_pack_iter_subset(self, f, subset, thin=None):
        if thin is None:
            thin = bool(list(self.store))
        resolve_ext_ref = thin and self.get_raw_no_repeat or None
        data = PackData('test.pack', file=f)
        assert data
        index = MemoryPackIndex.for_pack(data)
        pack = Pack.from_objects(data, index)
        return TestPackIterator.for_pack_subset(pack, subset, resolve_ext_ref=resolve_ext_ref)

    def assertEntriesMatch(self, expected_indexes, entries, pack_iter):
        expected = [entries[i] for i in expected_indexes]
        self.assertEqual(expected, list(pack_iter._walk_all_chains()))

    def test_no_deltas(self):
        f = BytesIO()
        entries = build_pack(f, [(Commit.type_num, b'commit'), (Blob.type_num, b'blob'), (Tree.type_num, b'tree')])
        self.assertEntriesMatch([0, 1, 2], entries, self.make_pack_iter(f))
        f.seek(0)
        self.assertEntriesMatch([], entries, self.make_pack_iter_subset(f, []))
        f.seek(0)
        self.assertEntriesMatch([1, 0], entries, self.make_pack_iter_subset(f, [entries[0][3], entries[1][3]]))
        f.seek(0)
        self.assertEntriesMatch([1, 0], entries, self.make_pack_iter_subset(f, [sha_to_hex(entries[0][3]), sha_to_hex(entries[1][3])]))

    def test_ofs_deltas(self):
        f = BytesIO()
        entries = build_pack(f, [(Blob.type_num, b'blob'), (OFS_DELTA, (0, b'blob1')), (OFS_DELTA, (0, b'blob2'))])
        self.assertEntriesMatch([0, 2, 1], entries, self.make_pack_iter(f))
        f.seek(0)
        self.assertEntriesMatch([0, 2, 1], entries, self.make_pack_iter_subset(f, [entries[1][3], entries[2][3]]))

    def test_ofs_deltas_chain(self):
        f = BytesIO()
        entries = build_pack(f, [(Blob.type_num, b'blob'), (OFS_DELTA, (0, b'blob1')), (OFS_DELTA, (1, b'blob2'))])
        self.assertEntriesMatch([0, 1, 2], entries, self.make_pack_iter(f))

    def test_ref_deltas(self):
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (1, b'blob1')), (Blob.type_num, b'blob'), (REF_DELTA, (1, b'blob2'))])
        self.assertEntriesMatch([1, 2, 0], entries, self.make_pack_iter(f))

    def test_ref_deltas_chain(self):
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (2, b'blob1')), (Blob.type_num, b'blob'), (REF_DELTA, (1, b'blob2'))])
        self.assertEntriesMatch([1, 2, 0], entries, self.make_pack_iter(f))

    def test_ofs_and_ref_deltas(self):
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (1, b'blob1')), (Blob.type_num, b'blob'), (OFS_DELTA, (1, b'blob2'))])
        self.assertEntriesMatch([1, 0, 2], entries, self.make_pack_iter(f))

    def test_mixed_chain(self):
        f = BytesIO()
        entries = build_pack(f, [(Blob.type_num, b'blob'), (REF_DELTA, (2, b'blob2')), (OFS_DELTA, (0, b'blob1')), (OFS_DELTA, (1, b'blob3')), (OFS_DELTA, (0, b'bob'))])
        self.assertEntriesMatch([0, 4, 2, 1, 3], entries, self.make_pack_iter(f))

    def test_long_chain(self):
        n = 100
        objects_spec = [(Blob.type_num, b'blob')]
        for i in range(n):
            objects_spec.append((OFS_DELTA, (i, b'blob' + str(i).encode('ascii'))))
        f = BytesIO()
        entries = build_pack(f, objects_spec)
        self.assertEntriesMatch(range(n + 1), entries, self.make_pack_iter(f))

    def test_branchy_chain(self):
        n = 100
        objects_spec = [(Blob.type_num, b'blob')]
        for i in range(n):
            objects_spec.append((OFS_DELTA, (0, b'blob' + str(i).encode('ascii'))))
        f = BytesIO()
        entries = build_pack(f, objects_spec)
        indices = [0, *list(range(100, 0, -1))]
        self.assertEntriesMatch(indices, entries, self.make_pack_iter(f))

    def test_ext_ref(self):
        blob, = self.store_blobs([b'blob'])
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (blob.id, b'blob1'))], store=self.store)
        pack_iter = self.make_pack_iter(f)
        self.assertEntriesMatch([0], entries, pack_iter)
        self.assertEqual([hex_to_sha(blob.id)], pack_iter.ext_refs())

    def test_ext_ref_chain(self):
        blob, = self.store_blobs([b'blob'])
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (1, b'blob2')), (REF_DELTA, (blob.id, b'blob1'))], store=self.store)
        pack_iter = self.make_pack_iter(f)
        self.assertEntriesMatch([1, 0], entries, pack_iter)
        self.assertEqual([hex_to_sha(blob.id)], pack_iter.ext_refs())

    def test_ext_ref_chain_degenerate(self):
        blob, = self.store_blobs([b'blob'])
        blob2, = self.store_blobs([b'blob2'])
        assert blob.id < blob2.id
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (blob.id, b'blob2')), (REF_DELTA, (0, b'blob3'))], store=self.store)
        pack_iter = self.make_pack_iter(f)
        self.assertEntriesMatch([0, 1], entries, pack_iter)
        self.assertEqual([hex_to_sha(blob.id)], pack_iter.ext_refs())

    def test_ext_ref_multiple_times(self):
        blob, = self.store_blobs([b'blob'])
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (blob.id, b'blob1')), (REF_DELTA, (blob.id, b'blob2'))], store=self.store)
        pack_iter = self.make_pack_iter(f)
        self.assertEntriesMatch([0, 1], entries, pack_iter)
        self.assertEqual([hex_to_sha(blob.id)], pack_iter.ext_refs())

    def test_multiple_ext_refs(self):
        b1, b2 = self.store_blobs([b'foo', b'bar'])
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (b1.id, b'foo1')), (REF_DELTA, (b2.id, b'bar2'))], store=self.store)
        pack_iter = self.make_pack_iter(f)
        self.assertEntriesMatch([0, 1], entries, pack_iter)
        self.assertEqual([hex_to_sha(b1.id), hex_to_sha(b2.id)], pack_iter.ext_refs())

    def test_bad_ext_ref_non_thin_pack(self):
        blob, = self.store_blobs([b'blob'])
        f = BytesIO()
        build_pack(f, [(REF_DELTA, (blob.id, b'blob1'))], store=self.store)
        pack_iter = self.make_pack_iter(f, thin=False)
        try:
            list(pack_iter._walk_all_chains())
            self.fail()
        except UnresolvedDeltas as e:
            self.assertEqual([blob.id], e.shas)

    def test_bad_ext_ref_thin_pack(self):
        b1, b2, b3 = self.store_blobs([b'foo', b'bar', b'baz'])
        f = BytesIO()
        build_pack(f, [(REF_DELTA, (1, b'foo99')), (REF_DELTA, (b1.id, b'foo1')), (REF_DELTA, (b2.id, b'bar2')), (REF_DELTA, (b3.id, b'baz3'))], store=self.store)
        del self.store[b2.id]
        del self.store[b3.id]
        pack_iter = self.make_pack_iter(f)
        try:
            list(pack_iter._walk_all_chains())
            self.fail()
        except UnresolvedDeltas as e:
            self.assertEqual((sorted([b2.id, b3.id]),), (sorted(e.shas),))