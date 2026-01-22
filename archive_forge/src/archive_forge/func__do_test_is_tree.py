from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def _do_test_is_tree(self, is_tree):
    self.assertFalse(is_tree(TreeEntry(None, None, None)))
    self.assertFalse(is_tree(TreeEntry(b'a', 33188, b'a' * 40)))
    self.assertFalse(is_tree(TreeEntry(b'a', 33261, b'a' * 40)))
    self.assertFalse(is_tree(TreeEntry(b'a', 40960, b'a' * 40)))
    self.assertTrue(is_tree(TreeEntry(b'a', 16384, b'a' * 40)))
    self.assertRaises(TypeError, is_tree, TreeEntry(b'a', b'x', b'a' * 40))
    self.assertRaises(AttributeError, is_tree, 1234)