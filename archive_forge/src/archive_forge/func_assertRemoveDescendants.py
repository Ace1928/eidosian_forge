from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def assertRemoveDescendants(self, expected, graph, revisions):
    parents = graph.get_parent_map(revisions)
    self.assertEqual(expected, graph._remove_simple_descendants(revisions, parents))