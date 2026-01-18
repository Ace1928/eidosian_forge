from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_reuse_detector(self):
    blob = make_object(Blob, data=b'blob')
    tree1 = self.commit_tree([(b'a', blob)])
    tree2 = self.commit_tree([(b'b', blob)])
    detector = RenameDetector(self.store)
    changes = [TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id))]
    self.assertEqual(changes, detector.changes_with_renames(tree1.id, tree2.id))
    self.assertEqual(changes, detector.changes_with_renames(tree1.id, tree2.id))