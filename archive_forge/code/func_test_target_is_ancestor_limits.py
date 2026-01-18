from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_target_is_ancestor_limits(self):
    """We shouldn't search all history if we run into ourselves"""
    graph = self.make_breaking_graph(ancestry_1, [b'rev1'])
    self.assertFindDistance(3, graph, b'rev3', [(b'rev4', 4)])