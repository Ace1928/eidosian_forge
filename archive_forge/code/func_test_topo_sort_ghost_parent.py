import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_topo_sort_ghost_parent(self):
    """Sort nodes, but don't include some parents in the output"""
    self.assertTopoSortOrder({0: [1], 1: [2]})