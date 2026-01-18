import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_cycle_2(self):
    """TopoSort traps graph with longer cycle"""
    self.assertSortAndIterateRaise(GraphCycleError, {0: [1], 1: [2], 2: [0]}.items())