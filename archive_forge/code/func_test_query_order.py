from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_query_order(self):
    pp1 = self.get_shared_provider(b'pp1', {b'a': ()}, has_cached=True)
    pp2 = self.get_shared_provider(b'pp2', {b'c': (b'b',)}, has_cached=False)
    pp3 = self.get_shared_provider(b'pp3', {b'b': (b'a',)}, has_cached=True)
    stacked = _mod_graph.StackedParentsProvider([pp1, pp2, pp3])
    self.assertEqual({b'a': (), b'b': (b'a',), b'c': (b'b',)}, stacked.get_parent_map([b'a', b'b', b'c', b'd']))
    self.assertEqual([(b'pp1', 'cached', [b'a', b'b', b'c', b'd']), (b'pp3', 'cached', [b'b', b'c', b'd']), (b'pp1', [b'c', b'd']), (b'pp2', [b'c', b'd']), (b'pp3', [b'd'])], self.calls)