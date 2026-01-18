from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_reverse_after_max_entries(self):
    c1, c2, c3 = self.make_linear_commits(3)
    self.assertWalkYields([c1, c2, c3], [c3.id], max_entries=3, reverse=True)
    self.assertWalkYields([c2, c3], [c3.id], max_entries=2, reverse=True)
    self.assertWalkYields([c3], [c3.id], max_entries=1, reverse=True)