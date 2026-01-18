import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_cycle(self):
    """TopoSort traps graph with cycles"""
    self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [0]}.items())