from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_octopus_no_conflict(self):
    r = list(range(5))
    blobs = [make_object(Blob, data=bytes(i)) for i in r]
    parents = [self.commit_tree([(b'a', blobs[i])]) for i in r]
    for i in r:
        self.assertChangesForMergeEqual([], parents, parents[i])