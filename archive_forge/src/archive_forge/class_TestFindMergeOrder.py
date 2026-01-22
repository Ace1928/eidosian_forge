from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestFindMergeOrder(TestGraphBase):

    def assertMergeOrder(self, expected, graph, tip, base_revisions):
        self.assertEqual(expected, graph.find_merge_order(tip, base_revisions))

    def test_parents(self):
        graph = self.make_graph(ancestry_1)
        self.assertMergeOrder([b'rev3', b'rev2b'], graph, b'rev4', [b'rev3', b'rev2b'])
        self.assertMergeOrder([b'rev3', b'rev2b'], graph, b'rev4', [b'rev2b', b'rev3'])

    def test_ancestors(self):
        graph = self.make_graph(ancestry_1)
        self.assertMergeOrder([b'rev1', b'rev2b'], graph, b'rev4', [b'rev1', b'rev2b'])
        self.assertMergeOrder([b'rev1', b'rev2b'], graph, b'rev4', [b'rev2b', b'rev1'])

    def test_shortcut_one_ancestor(self):
        graph = self.make_breaking_graph(ancestry_1, [b'rev3', b'rev2b', b'rev4'])
        self.assertMergeOrder([b'rev3'], graph, b'rev4', [b'rev3'])

    def test_shortcut_after_one_ancestor(self):
        graph = self.make_breaking_graph(ancestry_1, [b'rev2a', b'rev2b'])
        self.assertMergeOrder([b'rev3', b'rev1'], graph, b'rev4', [b'rev1', b'rev3'])