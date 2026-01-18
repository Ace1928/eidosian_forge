from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_unique_lca_null_revision2(self):
    """Ensure we pick NULL_REVISION when necessary"""
    graph = self.make_graph(ancestry_2)
    self.assertEqual(NULL_REVISION, graph.find_unique_lca(b'rev4a', b'rev1b'))