from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
class TestSearchResultRefine(tests.TestCase):

    def make_graph(self, ancestors):
        return _mod_graph.Graph(_mod_graph.DictParentsProvider(ancestors))

    def test_refine(self):
        self.make_graph({b'tip': [b'mid'], b'mid': [b'base'], b'tag': [b'base'], b'base': [NULL_REVISION], NULL_REVISION: []})
        result = vf_search.SearchResult({b'tip', b'tag'}, {NULL_REVISION}, 4, {b'tip', b'mid', b'tag', b'base'})
        result = result.refine({b'tip'}, {b'mid'})
        recipe = result.get_recipe()
        self.assertEqual({b'mid', b'tag'}, recipe[1])
        self.assertEqual({NULL_REVISION, b'tip'}, recipe[2])
        self.assertEqual(3, recipe[3])
        result = result.refine({b'mid', b'tag', b'base'}, {NULL_REVISION})
        recipe = result.get_recipe()
        self.assertEqual(set(), recipe[1])
        self.assertEqual({NULL_REVISION, b'tip', b'tag', b'mid'}, recipe[2])
        self.assertEqual(0, recipe[3])
        self.assertTrue(result.is_empty())