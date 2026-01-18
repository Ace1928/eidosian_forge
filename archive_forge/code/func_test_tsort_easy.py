import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_easy(self):
    """TopoSort list with one node"""
    self.assertSortAndIterate({0: []}.items(), [0])