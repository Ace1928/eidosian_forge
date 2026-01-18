import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
def test_stable_sorting(self):
    self.assertSorted(['b', 'c', 'a'], {'a': (), 'b': ('a',), 'c': ('a',)})
    self.assertSorted(['b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',)})
    self.assertSorted(['b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',)})
    self.assertSorted(['Z', 'b', 'c', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',), 'Z': ('a',)})
    self.assertSorted(['e', 'b', 'c', 'f', 'Z', 'd', 'a'], {'a': (), 'b': ('a',), 'c': ('a',), 'd': ('a',), 'Z': ('a',), 'e': ('b', 'c', 'd'), 'f': ('d', 'Z')})