import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def assertSortAndIterateRaise(self, exception_type, graph):
    """Try iterating and topo_sorting graph and expect an exception."""
    self.assertRaises(exception_type, topo_sort, graph)
    self.assertRaises(exception_type, list, TopoSorter(graph).iter_topo_order())