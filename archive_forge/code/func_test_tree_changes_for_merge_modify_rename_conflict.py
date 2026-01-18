from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_modify_rename_conflict(self):
    blob1 = make_object(Blob, data=b'a\nb\nc\nd\n')
    blob2 = make_object(Blob, data=b'a\nb\nc\ne\n')
    parent1 = self.commit_tree([(b'a', blob1)])
    parent2 = self.commit_tree([(b'b', blob1)])
    merge = self.commit_tree([(b'b', blob2)])
    self.assertChangesForMergeEqual([[TreeChange(CHANGE_RENAME, (b'a', F, blob1.id), (b'b', F, blob2.id)), TreeChange(CHANGE_MODIFY, (b'b', F, blob1.id), (b'b', F, blob2.id))]], [parent1, parent2], merge, rename_detector=self.detector)