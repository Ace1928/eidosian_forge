from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestCollapseLinearRegions(tests.TestCase):

    def assertCollapsed(self, collapsed, original):
        self.assertEqual(collapsed, _mod_graph.collapse_linear_regions(original))

    def test_collapse_nothing(self):
        d = {1: [2, 3], 2: [], 3: []}
        self.assertCollapsed(d, d)
        d = {1: [2], 2: [3, 4], 3: [5], 4: [5], 5: []}
        self.assertCollapsed(d, d)

    def test_collapse_chain(self):
        d = {1: [2], 2: [3], 3: [4], 4: [5], 5: []}
        self.assertCollapsed({1: [5], 5: []}, d)
        d = {5: [4], 4: [3], 3: [2], 2: [1], 1: []}
        self.assertCollapsed({5: [1], 1: []}, d)
        d = {5: [3], 3: [4], 4: [1], 1: [2], 2: []}
        self.assertCollapsed({5: [2], 2: []}, d)

    def test_collapse_with_multiple_children(self):
        d = {1: [2, 3], 2: [4], 4: [6], 3: [5], 5: [6], 6: [7], 7: []}
        self.assertCollapsed(d, d)