from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_add_exact_rename_conflict(self):
    blob = make_object(Blob, data=b'a\nb\nc\nd\n')
    parent1 = self.commit_tree([(b'a', blob)])
    parent2 = self.commit_tree([])
    merge = self.commit_tree([(b'b', blob)])
    self.assertChangesForMergeEqual([[TreeChange(CHANGE_RENAME, (b'a', F, blob.id), (b'b', F, blob.id)), TreeChange.add((b'b', F, blob.id))]], [parent1, parent2], merge, rename_detector=self.detector)