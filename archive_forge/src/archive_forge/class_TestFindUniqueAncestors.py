from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestFindUniqueAncestors(TestGraphBase):

    def assertFindUniqueAncestors(self, graph, expected, node, common):
        actual = graph.find_unique_ancestors(node, common)
        self.assertEqual(expected, sorted(actual))

    def test_empty_set(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindUniqueAncestors(graph, [], b'rev1', [b'rev1'])
        self.assertFindUniqueAncestors(graph, [], b'rev2b', [b'rev2b'])
        self.assertFindUniqueAncestors(graph, [], b'rev3', [b'rev1', b'rev3'])

    def test_single_node(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindUniqueAncestors(graph, [b'rev2a'], b'rev2a', [b'rev1'])
        self.assertFindUniqueAncestors(graph, [b'rev2b'], b'rev2b', [b'rev1'])
        self.assertFindUniqueAncestors(graph, [b'rev3'], b'rev3', [b'rev2a'])

    def test_minimal_ancestry(self):
        graph = self.make_breaking_graph(extended_history_shortcut, [NULL_REVISION, b'a', b'b'])
        self.assertFindUniqueAncestors(graph, [b'e'], b'e', [b'd'])
        graph = self.make_breaking_graph(extended_history_shortcut, [b'b'])
        self.assertFindUniqueAncestors(graph, [b'f'], b'f', [b'a', b'd'])
        graph = self.make_breaking_graph(complex_shortcut, [b'a', b'b'])
        self.assertFindUniqueAncestors(graph, [b'h'], b'h', [b'i'])
        self.assertFindUniqueAncestors(graph, [b'e', b'g', b'i'], b'i', [b'h'])
        self.assertFindUniqueAncestors(graph, [b'h'], b'h', [b'g'])
        self.assertFindUniqueAncestors(graph, [b'h'], b'h', [b'j'])

    def test_in_ancestry(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindUniqueAncestors(graph, [], b'rev1', [b'rev3'])
        self.assertFindUniqueAncestors(graph, [], b'rev2b', [b'rev4'])

    def test_multiple_revisions(self):
        graph = self.make_graph(ancestry_1)
        self.assertFindUniqueAncestors(graph, [b'rev4'], b'rev4', [b'rev3', b'rev2b'])
        self.assertFindUniqueAncestors(graph, [b'rev2a', b'rev3', b'rev4'], b'rev4', [b'rev2b'])

    def test_complex_shortcut(self):
        graph = self.make_graph(complex_shortcut)
        self.assertFindUniqueAncestors(graph, [b'h', b'n'], b'n', [b'm'])
        self.assertFindUniqueAncestors(graph, [b'e', b'i', b'm'], b'm', [b'n'])

    def test_complex_shortcut2(self):
        graph = self.make_graph(complex_shortcut2)
        self.assertFindUniqueAncestors(graph, [b'j', b'u'], b'u', [b't'])
        self.assertFindUniqueAncestors(graph, [b't'], b't', [b'u'])

    def test_multiple_interesting_unique(self):
        graph = self.make_graph(multiple_interesting_unique)
        self.assertFindUniqueAncestors(graph, [b'j', b'y'], b'y', [b'z'])
        self.assertFindUniqueAncestors(graph, [b'p', b'z'], b'z', [b'y'])

    def test_racing_shortcuts(self):
        graph = self.make_graph(racing_shortcuts)
        self.assertFindUniqueAncestors(graph, [b'p', b'q', b'z'], b'z', [b'y'])
        self.assertFindUniqueAncestors(graph, [b'h', b'i', b'j', b'y'], b'j', [b'z'])