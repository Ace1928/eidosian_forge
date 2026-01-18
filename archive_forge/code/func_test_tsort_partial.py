import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_tsort_partial(self):
    """Topological sort with partial ordering.

        Multiple correct orderings are possible, so test for
        correctness, not for exact match on the resulting list.
        """
    self.assertSortAndIterateOrder([(0, []), (1, [0]), (2, [0]), (3, [0]), (4, [1, 2, 3]), (5, [1, 2]), (6, [1, 2]), (7, [2, 3]), (8, [0, 1, 4, 5, 6])])