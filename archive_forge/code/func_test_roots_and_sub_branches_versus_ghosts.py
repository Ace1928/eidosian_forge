import pprint
from breezy.errors import GraphCycleError
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase
from breezy.tsort import MergeSorter, TopoSorter, merge_sort, topo_sort
def test_roots_and_sub_branches_versus_ghosts(self):
    """Extra roots and their mini branches use the same numbering.

        All of them use the 0-node numbering.
        """
    self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B'], 'D': [], 'E': ['D'], 'F': ['D'], 'G': ['E', 'F'], 'H': ['C', 'G'], 'I': [], 'J': ['H', 'I'], 'K': [], 'L': ['K'], 'M': ['K'], 'N': ['L', 'M'], 'O': ['N'], 'P': ['N'], 'Q': ['O', 'P'], 'R': ['J', 'Q']}.items(), 'R', [(0, 'R', 0, (6,), False), (1, 'Q', 1, (0, 4, 5), False), (2, 'P', 2, (0, 6, 1), True), (3, 'O', 1, (0, 4, 4), False), (4, 'N', 1, (0, 4, 3), False), (5, 'M', 2, (0, 5, 1), True), (6, 'L', 1, (0, 4, 2), False), (7, 'K', 1, (0, 4, 1), True), (8, 'J', 0, (5,), False), (9, 'I', 1, (0, 3, 1), True), (10, 'H', 0, (4,), False), (11, 'G', 1, (0, 1, 3), False), (12, 'F', 2, (0, 2, 1), True), (13, 'E', 1, (0, 1, 2), False), (14, 'D', 1, (0, 1, 1), True), (15, 'C', 0, (3,), False), (16, 'B', 0, (2,), False), (17, 'A', 0, (1,), True)], True)