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
class TreeLookupPathTests(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.store = MemoryObjectStore()
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        blob_c = make_object(Blob, data=b'c')
        for blob in [blob_a, blob_b, blob_c]:
            self.store.add_object(blob)
        blobs = [(b'a', blob_a.id, 33188), (b'ad/b', blob_b.id, 33188), (b'ad/bd/c', blob_c.id, 33261), (b'ad/c', blob_c.id, 33188), (b'c', blob_c.id, 33188), (b'd', blob_c.id, S_IFGITLINK)]
        self.tree_id = commit_tree(self.store, blobs)

    def get_object(self, sha):
        return self.store[sha]

    def test_lookup_blob(self):
        o_id = tree_lookup_path(self.get_object, self.tree_id, b'a')[1]
        self.assertIsInstance(self.store[o_id], Blob)

    def test_lookup_tree(self):
        o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad')[1]
        self.assertIsInstance(self.store[o_id], Tree)
        o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad/bd')[1]
        self.assertIsInstance(self.store[o_id], Tree)
        o_id = tree_lookup_path(self.get_object, self.tree_id, b'ad/bd/')[1]
        self.assertIsInstance(self.store[o_id], Tree)

    def test_lookup_submodule(self):
        tree_lookup_path(self.get_object, self.tree_id, b'd')[1]
        self.assertRaises(SubmoduleEncountered, tree_lookup_path, self.get_object, self.tree_id, b'd/a')

    def test_lookup_nonexistent(self):
        self.assertRaises(KeyError, tree_lookup_path, self.get_object, self.tree_id, b'j')

    def test_lookup_not_tree(self):
        self.assertRaises(NotTreeError, tree_lookup_path, self.get_object, self.tree_id, b'ad/b/j')