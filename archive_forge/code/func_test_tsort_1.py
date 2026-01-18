import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_1(self):
    """TopoSort simple nontrivial graph"""
    self.assertSortAndIterate({0: [3], 1: [4], 2: [1, 4], 3: [], 4: [0, 3]}.items(), [3, 0, 4, 1, 2])