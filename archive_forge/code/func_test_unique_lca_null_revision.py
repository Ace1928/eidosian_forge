from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_unique_lca_null_revision(self):
    """Ensure we pick NULL_REVISION when necessary"""
    graph = self.make_graph(criss_cross2)
    self.assertEqual(b'rev1b', graph.find_unique_lca(b'rev2a', b'rev1b'))
    self.assertEqual(NULL_REVISION, graph.find_unique_lca(b'rev2a', b'rev2b'))