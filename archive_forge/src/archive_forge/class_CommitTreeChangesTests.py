import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
class CommitTreeChangesTests(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.blob_a = make_object(Blob, data=b'a')
        self.blob_b = make_object(Blob, data=b'b')
        self.blob_c = make_object(Blob, data=b'c')
        for blob in [self.blob_a, self.blob_b, self.blob_c]:
            self.store.add_object(blob)
        blobs = [(b'a', self.blob_a.id, 33188), (b'ad/b', self.blob_b.id, 33188), (b'ad/bd/c', self.blob_c.id, 33261), (b'ad/c', self.blob_c.id, 33188), (b'c', self.blob_c.id, 33188)]
        self.tree_id = commit_tree(self.store, blobs)

    def test_no_changes(self):
        self.assertEqual(self.store[self.tree_id], commit_tree_changes(self.store, self.store[self.tree_id], []))

    def test_add_blob(self):
        blob_d = make_object(Blob, data=b'd')
        new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'd', 33188, blob_d.id)])
        self.assertEqual(new_tree[b'd'], (33188, b'c59d9b6344f1af00e504ba698129f07a34bbed8d'))

    def test_add_blob_in_dir(self):
        blob_d = make_object(Blob, data=b'd')
        new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'e/f/d', 33188, blob_d.id)])
        self.assertEqual(new_tree.items(), [TreeEntry(path=b'a', mode=stat.S_IFREG | 33188, sha=self.blob_a.id), TreeEntry(path=b'ad', mode=stat.S_IFDIR, sha=b'0e2ce2cd7725ff4817791be31ccd6e627e801f4a'), TreeEntry(path=b'c', mode=stat.S_IFREG | 33188, sha=self.blob_c.id), TreeEntry(path=b'e', mode=stat.S_IFDIR, sha=b'6ab344e288724ac2fb38704728b8896e367ed108')])
        e_tree = self.store[new_tree[b'e'][1]]
        self.assertEqual(e_tree.items(), [TreeEntry(path=b'f', mode=stat.S_IFDIR, sha=b'24d2c94d8af232b15a0978c006bf61ef4479a0a5')])
        f_tree = self.store[e_tree[b'f'][1]]
        self.assertEqual(f_tree.items(), [TreeEntry(path=b'd', mode=stat.S_IFREG | 33188, sha=blob_d.id)])

    def test_delete_blob(self):
        new_tree = commit_tree_changes(self.store, self.store[self.tree_id], [(b'ad/bd/c', None, None)])
        self.assertEqual(set(new_tree), {b'a', b'ad', b'c'})
        ad_tree = self.store[new_tree[b'ad'][1]]
        self.assertEqual(set(ad_tree), {b'b', b'c'})