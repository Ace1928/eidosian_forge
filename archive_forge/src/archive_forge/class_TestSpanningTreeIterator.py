import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestSpanningTreeIterator:
    """
    Tests the spanning tree iterator on the example graph in the 2005 Sörensen
    and Janssens paper An Algorithm to Generate all Spanning Trees of a Graph in
    Order of Increasing Cost
    """

    def setup_method(self):
        edges = [(0, 1, 5), (1, 2, 4), (1, 4, 6), (2, 3, 5), (2, 4, 7), (3, 4, 3)]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        self.spanning_trees = [[(0, 1, {'weight': 5}), (1, 2, {'weight': 4}), (2, 3, {'weight': 5}), (3, 4, {'weight': 3})], [(0, 1, {'weight': 5}), (1, 2, {'weight': 4}), (1, 4, {'weight': 6}), (3, 4, {'weight': 3})], [(0, 1, {'weight': 5}), (1, 4, {'weight': 6}), (2, 3, {'weight': 5}), (3, 4, {'weight': 3})], [(0, 1, {'weight': 5}), (1, 2, {'weight': 4}), (2, 4, {'weight': 7}), (3, 4, {'weight': 3})], [(0, 1, {'weight': 5}), (1, 2, {'weight': 4}), (1, 4, {'weight': 6}), (2, 3, {'weight': 5})], [(0, 1, {'weight': 5}), (1, 4, {'weight': 6}), (2, 4, {'weight': 7}), (3, 4, {'weight': 3})], [(0, 1, {'weight': 5}), (1, 2, {'weight': 4}), (2, 3, {'weight': 5}), (2, 4, {'weight': 7})], [(0, 1, {'weight': 5}), (1, 4, {'weight': 6}), (2, 3, {'weight': 5}), (2, 4, {'weight': 7})]]

    def test_minimum_spanning_tree_iterator(self):
        """
        Tests that the spanning trees are correctly returned in increasing order
        """
        tree_index = 0
        for tree in nx.SpanningTreeIterator(self.G):
            actual = sorted(tree.edges(data=True))
            assert edges_equal(actual, self.spanning_trees[tree_index])
            tree_index += 1

    def test_maximum_spanning_tree_iterator(self):
        """
        Tests that the spanning trees are correctly returned in decreasing order
        """
        tree_index = 7
        for tree in nx.SpanningTreeIterator(self.G, minimum=False):
            actual = sorted(tree.edges(data=True))
            assert edges_equal(actual, self.spanning_trees[tree_index])
            tree_index -= 1