from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_to_tree(self):
    blob_a = make_object(Blob, data=b'a')
    blob_x = make_object(Blob, data=b'x')
    tree1 = self.commit_tree([(b'a', blob_a)])
    tree2 = self.commit_tree([(b'a/x', blob_x)])
    self.assertChangesEqual([TreeChange.delete((b'a', F, blob_a.id)), TreeChange.add((b'a/x', F, blob_x.id))], tree1, tree2)