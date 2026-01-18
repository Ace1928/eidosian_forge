from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
def test_refine(self):
    g = self.make_graph({b'tip': [b'mid'], b'mid': [b'base'], b'tag': [b'base'], b'base': [NULL_REVISION], NULL_REVISION: []})
    result = vf_search.PendingAncestryResult([b'tip', b'tag'], None)
    result = result.refine({b'tip'}, {b'mid'})
    self.assertEqual({b'mid', b'tag'}, result.heads)
    result = result.refine({b'mid', b'tag', b'base'}, {NULL_REVISION})
    self.assertEqual({NULL_REVISION}, result.heads)
    self.assertTrue(result.is_empty())