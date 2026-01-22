from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def assertChangesEqual(self, expected, tree1, tree2, **kwargs):
    actual = list(tree_changes(self.store, tree1.id, tree2.id, **kwargs))
    self.assertEqual(expected, actual)