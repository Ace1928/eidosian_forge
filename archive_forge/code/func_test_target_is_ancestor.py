from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_target_is_ancestor(self):
    graph = self.make_graph(ancestry_1)
    self.assertFindDistance(2, graph, b'rev2a', [(b'rev3', 3)])