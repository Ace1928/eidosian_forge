from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_rev_is_ghost(self):
    graph = self.make_graph(ancestry_1)
    e = self.assertRaises(errors.GhostRevisionsHaveNoRevno, graph.find_distance_to_null, b'rev_missing', [])
    self.assertEqual(b'rev_missing', e.revision_id)
    self.assertEqual(b'rev_missing', e.ghost_revision_id)