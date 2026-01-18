import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_topo_sort_cycle(self):
    """TopoSort traps graph with cycles"""
    g = self.make_known_graph({0: [1], 1: [0]})
    self.assertRaises(errors.GraphCycleError, g.topo_sort)