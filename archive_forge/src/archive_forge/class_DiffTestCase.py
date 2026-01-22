from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
class DiffTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()
        self.empty_tree = self.commit_tree([])

    def commit_tree(self, entries):
        commit_blobs = []
        for entry in entries:
            if len(entry) == 2:
                path, obj = entry
                mode = F
            else:
                path, obj, mode = entry
            if isinstance(obj, Blob):
                self.store.add_object(obj)
                sha = obj.id
            else:
                sha = obj
            commit_blobs.append((path, sha, mode))
        return self.store[commit_tree(self.store, commit_blobs)]