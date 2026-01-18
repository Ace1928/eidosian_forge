from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_octopus_delete(self):
    blob1 = make_object(Blob, data=b'1')
    blob2 = make_object(Blob, data=b'3')
    parent1 = self.commit_tree([(b'a', blob1)])
    parent2 = self.commit_tree([(b'a', blob2)])
    parent3 = merge = self.commit_tree([])
    self.assertChangesForMergeEqual([], [parent1, parent1, parent1], merge)
    self.assertChangesForMergeEqual([], [parent1, parent1, parent3], merge)
    self.assertChangesForMergeEqual([], [parent1, parent3, parent3], merge)
    self.assertChangesForMergeEqual([[TreeChange.delete((b'a', F, blob1.id)), TreeChange.delete((b'a', F, blob2.id)), None]], [parent1, parent2, parent3], merge)