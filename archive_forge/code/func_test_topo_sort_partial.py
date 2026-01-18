import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_topo_sort_partial(self):
    """Topological sort with partial ordering.

        Multiple correct orderings are possible, so test for
        correctness, not for exact match on the resulting list.
        """
    self.assertTopoSortOrder({0: [], 1: [0], 2: [0], 3: [0], 4: [1, 2, 3], 5: [1, 2], 6: [1, 2], 7: [2, 3], 8: [0, 1, 4, 5, 6]})