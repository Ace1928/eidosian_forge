from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_modify_mode(self):
    blob_a = make_object(Blob, data=b'a')
    tree1 = self.commit_tree([(b'a', blob_a, 33188)])
    tree2 = self.commit_tree([(b'a', blob_a, 33261)])
    self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', 33188, blob_a.id), (b'a', 33261, blob_a.id))], tree1, tree2)