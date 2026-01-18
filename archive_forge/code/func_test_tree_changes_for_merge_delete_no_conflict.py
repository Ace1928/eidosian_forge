from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_for_merge_delete_no_conflict(self):
    blob = make_object(Blob, data=b'blob')
    has = self.commit_tree([(b'a', blob)])
    doesnt_have = self.commit_tree([])
    self.assertChangesForMergeEqual([], [has, has], doesnt_have)
    self.assertChangesForMergeEqual([], [has, doesnt_have], doesnt_have)