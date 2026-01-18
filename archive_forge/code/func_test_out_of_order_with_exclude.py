from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def test_out_of_order_with_exclude(self):
    c1, x2, y3, y4, y5, m6 = self.make_commits([[1], [2, 1], [3, 1], [4, 3], [5, 4], [6, 2, 4]], times=[2, 3, 4, 5, 1, 6])
    self.assertWalkYields([m6, y4, y3, x2, c1], [m6.id])
    self.assertWalkYields([m6, x2], [m6.id], exclude=[y5.id])