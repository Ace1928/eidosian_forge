from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestStackedParentsProvider(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.calls = []

    def get_shared_provider(self, info, ancestry, has_cached):
        pp = _mod_graph.DictParentsProvider(ancestry)
        if has_cached:
            pp.get_cached_parent_map = pp.get_parent_map
        return SharedInstrumentedParentsProvider(pp, self.calls, info)

    def test_stacked_parents_provider(self):
        parents1 = _mod_graph.DictParentsProvider({b'rev2': [b'rev3']})
        parents2 = _mod_graph.DictParentsProvider({b'rev1': [b'rev4']})
        stacked = _mod_graph.StackedParentsProvider([parents1, parents2])
        self.assertEqual({b'rev1': [b'rev4'], b'rev2': [b'rev3']}, stacked.get_parent_map([b'rev1', b'rev2']))
        self.assertEqual({b'rev2': [b'rev3'], b'rev1': [b'rev4']}, stacked.get_parent_map([b'rev2', b'rev1']))
        self.assertEqual({b'rev2': [b'rev3']}, stacked.get_parent_map([b'rev2', b'rev2']))
        self.assertEqual({b'rev1': [b'rev4']}, stacked.get_parent_map([b'rev1', b'rev1']))

    def test_stacked_parents_provider_overlapping(self):
        parents1 = _mod_graph.DictParentsProvider({b'rev2': [b'rev1']})
        parents2 = _mod_graph.DictParentsProvider({b'rev2': [b'rev1']})
        stacked = _mod_graph.StackedParentsProvider([parents1, parents2])
        self.assertEqual({b'rev2': [b'rev1']}, stacked.get_parent_map([b'rev2']))

    def test_handles_no_get_cached_parent_map(self):
        pp1 = self.get_shared_provider(b'pp1', {b'rev2': (b'rev1',)}, has_cached=False)
        pp2 = self.get_shared_provider(b'pp2', {b'rev2': (b'rev1',)}, has_cached=True)
        stacked = _mod_graph.StackedParentsProvider([pp1, pp2])
        self.assertEqual({b'rev2': (b'rev1',)}, stacked.get_parent_map([b'rev2']))
        self.assertEqual([(b'pp2', 'cached', [b'rev2'])], self.calls)

    def test_query_order(self):
        pp1 = self.get_shared_provider(b'pp1', {b'a': ()}, has_cached=True)
        pp2 = self.get_shared_provider(b'pp2', {b'c': (b'b',)}, has_cached=False)
        pp3 = self.get_shared_provider(b'pp3', {b'b': (b'a',)}, has_cached=True)
        stacked = _mod_graph.StackedParentsProvider([pp1, pp2, pp3])
        self.assertEqual({b'a': (), b'b': (b'a',), b'c': (b'b',)}, stacked.get_parent_map([b'a', b'b', b'c', b'd']))
        self.assertEqual([(b'pp1', 'cached', [b'a', b'b', b'c', b'd']), (b'pp3', 'cached', [b'b', b'c', b'd']), (b'pp1', [b'c', b'd']), (b'pp2', [b'c', b'd']), (b'pp3', [b'd'])], self.calls)