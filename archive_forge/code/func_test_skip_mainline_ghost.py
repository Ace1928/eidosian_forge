import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_skip_mainline_ghost(self):
    self.assertSorted(['b', 'c', 'a'], {'a': (), 'b': ('ghost', 'a'), 'c': ('a',)})