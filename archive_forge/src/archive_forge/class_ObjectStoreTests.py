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
class ObjectStoreTests:

    def test_determine_wants_all(self):
        self.assertEqual([b'1' * 40], self.store.determine_wants_all({b'refs/heads/foo': b'1' * 40}))

    def test_determine_wants_all_zero(self):
        self.assertEqual([], self.store.determine_wants_all({b'refs/heads/foo': b'0' * 40}))

    @skipUnless(patch, 'Required mock.patch')
    def test_determine_wants_all_depth(self):
        self.store.add_object(testobject)
        refs = {b'refs/heads/foo': testobject.id}
        with patch.object(self.store, '_get_depth', return_value=1) as m:
            self.assertEqual([], self.store.determine_wants_all(refs, depth=0))
            self.assertEqual([testobject.id], self.store.determine_wants_all(refs, depth=DEPTH_INFINITE))
            m.assert_not_called()
            self.assertEqual([], self.store.determine_wants_all(refs, depth=1))
            m.assert_called_with(testobject.id)
            self.assertEqual([testobject.id], self.store.determine_wants_all(refs, depth=2))

    def test_get_depth(self):
        self.assertEqual(0, self.store._get_depth(testobject.id))
        self.store.add_object(testobject)
        self.assertEqual(1, self.store._get_depth(testobject.id, get_parents=lambda x: []))
        parent = make_object(Blob, data=b'parent data')
        self.store.add_object(parent)
        self.assertEqual(2, self.store._get_depth(testobject.id, get_parents=lambda x: [parent.id] if x == testobject else []))

    def test_iter(self):
        self.assertEqual([], list(self.store))

    def test_get_nonexistant(self):
        self.assertRaises(KeyError, lambda: self.store[b'a' * 40])

    def test_contains_nonexistant(self):
        self.assertNotIn(b'a' * 40, self.store)

    def test_add_objects_empty(self):
        self.store.add_objects([])

    def test_add_commit(self):
        self.store.add_objects([])

    def test_store_resilience(self):
        """Test if updating an existing stored object doesn't erase the
        object from the store.
        """
        test_object = make_object(Blob, data=b'data')
        self.store.add_object(test_object)
        test_object_id = test_object.id
        test_object.data = test_object.data + b'update'
        stored_test_object = self.store[test_object_id]
        self.assertNotEqual(test_object.id, stored_test_object.id)
        self.assertEqual(stored_test_object.id, test_object_id)

    def test_add_object(self):
        self.store.add_object(testobject)
        self.assertEqual({testobject.id}, set(self.store))
        self.assertIn(testobject.id, self.store)
        r = self.store[testobject.id]
        self.assertEqual(r, testobject)

    def test_add_objects(self):
        data = [(testobject, 'mypath')]
        self.store.add_objects(data)
        self.assertEqual({testobject.id}, set(self.store))
        self.assertIn(testobject.id, self.store)
        r = self.store[testobject.id]
        self.assertEqual(r, testobject)

    def test_tree_changes(self):
        blob_a1 = make_object(Blob, data=b'a1')
        blob_a2 = make_object(Blob, data=b'a2')
        blob_b = make_object(Blob, data=b'b')
        for blob in [blob_a1, blob_a2, blob_b]:
            self.store.add_object(blob)
        blobs_1 = [(b'a', blob_a1.id, 33188), (b'b', blob_b.id, 33188)]
        tree1_id = commit_tree(self.store, blobs_1)
        blobs_2 = [(b'a', blob_a2.id, 33188), (b'b', blob_b.id, 33188)]
        tree2_id = commit_tree(self.store, blobs_2)
        change_a = ((b'a', b'a'), (33188, 33188), (blob_a1.id, blob_a2.id))
        self.assertEqual([change_a], list(self.store.tree_changes(tree1_id, tree2_id)))
        self.assertEqual([change_a, ((b'b', b'b'), (33188, 33188), (blob_b.id, blob_b.id))], list(self.store.tree_changes(tree1_id, tree2_id, want_unchanged=True)))

    def test_iter_tree_contents(self):
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        blob_c = make_object(Blob, data=b'c')
        for blob in [blob_a, blob_b, blob_c]:
            self.store.add_object(blob)
        blobs = [(b'a', blob_a.id, 33188), (b'ad/b', blob_b.id, 33188), (b'ad/bd/c', blob_c.id, 33261), (b'ad/c', blob_c.id, 33188), (b'c', blob_c.id, 33188)]
        tree_id = commit_tree(self.store, blobs)
        self.assertEqual([TreeEntry(p, m, h) for p, h, m in blobs], list(iter_tree_contents(self.store, tree_id)))
        self.assertEqual([], list(iter_tree_contents(self.store, None)))

    def test_iter_tree_contents_include_trees(self):
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        blob_c = make_object(Blob, data=b'c')
        for blob in [blob_a, blob_b, blob_c]:
            self.store.add_object(blob)
        blobs = [(b'a', blob_a.id, 33188), (b'ad/b', blob_b.id, 33188), (b'ad/bd/c', blob_c.id, 33261)]
        tree_id = commit_tree(self.store, blobs)
        tree = self.store[tree_id]
        tree_ad = self.store[tree[b'ad'][1]]
        tree_bd = self.store[tree_ad[b'bd'][1]]
        expected = [TreeEntry(b'', 16384, tree_id), TreeEntry(b'a', 33188, blob_a.id), TreeEntry(b'ad', 16384, tree_ad.id), TreeEntry(b'ad/b', 33188, blob_b.id), TreeEntry(b'ad/bd', 16384, tree_bd.id), TreeEntry(b'ad/bd/c', 33261, blob_c.id)]
        actual = iter_tree_contents(self.store, tree_id, include_trees=True)
        self.assertEqual(expected, list(actual))

    def make_tag(self, name, obj):
        tag = make_tag(obj, name=name)
        self.store.add_object(tag)
        return tag

    def test_peel_sha(self):
        self.store.add_object(testobject)
        tag1 = self.make_tag(b'1', testobject)
        tag2 = self.make_tag(b'2', testobject)
        tag3 = self.make_tag(b'3', testobject)
        for obj in [testobject, tag1, tag2, tag3]:
            self.assertEqual((obj, testobject), peel_sha(self.store, obj.id))

    def test_get_raw(self):
        self.store.add_object(testobject)
        self.assertEqual((Blob.type_num, b'yummy data'), self.store.get_raw(testobject.id))

    def test_close(self):
        self.store.add_object(testobject)
        self.store.close()