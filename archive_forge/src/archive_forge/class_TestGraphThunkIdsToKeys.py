from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestGraphThunkIdsToKeys(tests.TestCase):

    def test_heads(self):
        d = {(b'D',): [(b'B',), (b'C',)], (b'C',): [(b'A',)], (b'B',): [(b'A',)], (b'A',): []}
        g = _mod_graph.Graph(_mod_graph.DictParentsProvider(d))
        graph_thunk = _mod_graph.GraphThunkIdsToKeys(g)
        self.assertEqual([b'D'], sorted(graph_thunk.heads([b'D', b'A'])))
        self.assertEqual([b'D'], sorted(graph_thunk.heads([b'D', b'B'])))
        self.assertEqual([b'D'], sorted(graph_thunk.heads([b'D', b'C'])))
        self.assertEqual([b'B', b'C'], sorted(graph_thunk.heads([b'B', b'C'])))

    def test_add_node(self):
        d = {(b'C',): [(b'A',)], (b'B',): [(b'A',)], (b'A',): []}
        g = _mod_graph.KnownGraph(d)
        graph_thunk = _mod_graph.GraphThunkIdsToKeys(g)
        graph_thunk.add_node(b'D', [b'A', b'C'])
        self.assertEqual([b'B', b'D'], sorted(graph_thunk.heads([b'D', b'B', b'A'])))

    def test_merge_sort(self):
        d = {(b'C',): [(b'A',)], (b'B',): [(b'A',)], (b'A',): []}
        g = _mod_graph.KnownGraph(d)
        graph_thunk = _mod_graph.GraphThunkIdsToKeys(g)
        graph_thunk.add_node(b'D', [b'A', b'C'])
        self.assertEqual([(b'C', 0, (2,), False), (b'A', 0, (1,), True)], [(n.key, n.merge_depth, n.revno, n.end_of_merge) for n in graph_thunk.merge_sort(b'C')])