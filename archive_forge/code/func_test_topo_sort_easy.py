import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_topo_sort_easy(self):
    """TopoSort list with one node"""
    self.assertTopoSortOrder({0: []})