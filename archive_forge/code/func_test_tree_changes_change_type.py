from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_change_type(self):
    blob_a1 = make_object(Blob, data=b'a')
    blob_a2 = make_object(Blob, data=b'/foo/bar')
    tree1 = self.commit_tree([(b'a', blob_a1, 33188)])
    tree2 = self.commit_tree([(b'a', blob_a2, 40960)])
    self.assertChangesEqual([TreeChange.delete((b'a', 33188, blob_a1.id)), TreeChange.add((b'a', 40960, blob_a2.id))], tree1, tree2)