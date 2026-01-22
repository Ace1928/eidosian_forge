import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestKnownGraph(TestCaseWithKnownGraph):

    def assertGDFO(self, graph, rev, gdfo):
        node = graph._nodes[rev]
        self.assertEqual(gdfo, node.gdfo)

    def test_children_ancestry1(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual([b'rev1'], graph.get_child_keys(NULL_REVISION))
        self.assertEqual([b'rev2a', b'rev2b'], sorted(graph.get_child_keys(b'rev1')))
        self.assertEqual([b'rev3'], graph.get_child_keys(b'rev2a'))
        self.assertEqual([b'rev4'], graph.get_child_keys(b'rev3'))
        self.assertEqual([b'rev4'], graph.get_child_keys(b'rev2b'))
        self.assertRaises(KeyError, graph.get_child_keys, b'not_in_graph')

    def test_parent_ancestry1(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual([NULL_REVISION], graph.get_parent_keys(b'rev1'))
        self.assertEqual([b'rev1'], graph.get_parent_keys(b'rev2a'))
        self.assertEqual([b'rev1'], graph.get_parent_keys(b'rev2b'))
        self.assertEqual([b'rev2a'], graph.get_parent_keys(b'rev3'))
        self.assertEqual([b'rev2b', b'rev3'], sorted(graph.get_parent_keys(b'rev4')))
        self.assertRaises(KeyError, graph.get_child_keys, b'not_in_graph')

    def test_parent_with_ghost(self):
        graph = self.make_known_graph(test_graph.with_ghost)
        self.assertEqual(None, graph.get_parent_keys(b'g'))

    def test_gdfo_ancestry_1(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertGDFO(graph, b'rev1', 2)
        self.assertGDFO(graph, b'rev2b', 3)
        self.assertGDFO(graph, b'rev2a', 3)
        self.assertGDFO(graph, b'rev3', 4)
        self.assertGDFO(graph, b'rev4', 5)

    def test_gdfo_feature_branch(self):
        graph = self.make_known_graph(test_graph.feature_branch)
        self.assertGDFO(graph, b'rev1', 2)
        self.assertGDFO(graph, b'rev2b', 3)
        self.assertGDFO(graph, b'rev3b', 4)

    def test_gdfo_extended_history_shortcut(self):
        graph = self.make_known_graph(test_graph.extended_history_shortcut)
        self.assertGDFO(graph, b'a', 2)
        self.assertGDFO(graph, b'b', 3)
        self.assertGDFO(graph, b'c', 4)
        self.assertGDFO(graph, b'd', 5)
        self.assertGDFO(graph, b'e', 6)
        self.assertGDFO(graph, b'f', 6)

    def test_gdfo_with_ghost(self):
        graph = self.make_known_graph(test_graph.with_ghost)
        self.assertGDFO(graph, b'f', 2)
        self.assertGDFO(graph, b'e', 3)
        self.assertGDFO(graph, b'g', 1)
        self.assertGDFO(graph, b'b', 4)
        self.assertGDFO(graph, b'd', 4)
        self.assertGDFO(graph, b'a', 5)
        self.assertGDFO(graph, b'c', 5)

    def test_add_existing_node(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertGDFO(graph, b'rev4', 5)
        graph.add_node(b'rev4', [b'rev3', b'rev2b'])
        self.assertGDFO(graph, b'rev4', 5)
        graph.add_node(b'rev4', (b'rev3', b'rev2b'))

    def test_add_existing_node_mismatched_parents(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertRaises(ValueError, graph.add_node, b'rev4', [b'rev2b', b'rev3'])

    def test_add_node_with_ghost_parent(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        graph.add_node(b'rev5', [b'rev2b', b'revGhost'])
        self.assertGDFO(graph, b'rev5', 4)
        self.assertGDFO(graph, b'revGhost', 1)

    def test_add_new_root(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        graph.add_node(b'rev5', [])
        self.assertGDFO(graph, b'rev5', 1)

    def test_add_with_all_ghost_parents(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        graph.add_node(b'rev5', [b'ghost'])
        self.assertGDFO(graph, b'rev5', 2)
        self.assertGDFO(graph, b'ghost', 1)

    def test_gdfo_after_add_node(self):
        graph = self.make_known_graph(test_graph.ancestry_1)
        self.assertEqual([], graph.get_child_keys(b'rev4'))
        graph.add_node(b'rev5', [b'rev4'])
        self.assertEqual([b'rev4'], graph.get_parent_keys(b'rev5'))
        self.assertEqual([b'rev5'], graph.get_child_keys(b'rev4'))
        self.assertEqual([], graph.get_child_keys(b'rev5'))
        self.assertGDFO(graph, b'rev5', 6)
        graph.add_node(b'rev6', [b'rev2b'])
        graph.add_node(b'rev7', [b'rev6'])
        graph.add_node(b'rev8', [b'rev7', b'rev5'])
        self.assertGDFO(graph, b'rev5', 6)
        self.assertGDFO(graph, b'rev6', 4)
        self.assertGDFO(graph, b'rev7', 5)
        self.assertGDFO(graph, b'rev8', 7)

    def test_fill_in_ghost(self):
        graph = self.make_known_graph(test_graph.with_ghost)
        graph.add_node(b'x', [])
        graph.add_node(b'y', [b'x'])
        graph.add_node(b'z', [b'y'])
        graph.add_node(b'g', [b'z'])
        self.assertGDFO(graph, b'f', 2)
        self.assertGDFO(graph, b'e', 3)
        self.assertGDFO(graph, b'x', 1)
        self.assertGDFO(graph, b'y', 2)
        self.assertGDFO(graph, b'z', 3)
        self.assertGDFO(graph, b'g', 4)
        self.assertGDFO(graph, b'b', 4)
        self.assertGDFO(graph, b'd', 5)
        self.assertGDFO(graph, b'a', 5)
        self.assertGDFO(graph, b'c', 6)