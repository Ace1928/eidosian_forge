from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_rename_threshold(self):
    blob1 = make_object(Blob, data=b'a\nb\nc\n')
    blob2 = make_object(Blob, data=b'a\nb\nd\n')
    tree1 = self.commit_tree([(b'a', blob1)])
    tree2 = self.commit_tree([(b'b', blob2)])
    self.assertEqual([TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rename_threshold=50))
    self.assertEqual([TreeChange.delete((b'a', F, blob1.id)), TreeChange.add((b'b', F, blob2.id))], self.detect_renames(tree1, tree2, rename_threshold=75))