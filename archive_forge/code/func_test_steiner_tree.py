import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
def test_steiner_tree(self):
    valid_steiner_trees = [[[(1, 2, {'weight': 10}), (2, 3, {'weight': 10}), (2, 7, {'weight': 1}), (3, 4, {'weight': 10}), (5, 7, {'weight': 1})], [(1, 2, {'weight': 10}), (2, 7, {'weight': 1}), (3, 4, {'weight': 10}), (4, 5, {'weight': 10}), (5, 7, {'weight': 1})], [(1, 2, {'weight': 10}), (2, 3, {'weight': 10}), (2, 7, {'weight': 1}), (4, 5, {'weight': 10}), (5, 7, {'weight': 1})]], [[(0, 5, {'weight': 6}), (1, 2, {'weight': 2}), (1, 5, {'weight': 3}), (3, 5, {'weight': 5})], [(0, 5, {'weight': 6}), (4, 2, {'weight': 4}), (4, 5, {'weight': 1}), (3, 5, {'weight': 5})]], [[(1, 10, {'weight': 2}), (3, 10, {'weight': 2}), (3, 11, {'weight': 1}), (5, 12, {'weight': 1}), (6, 13, {'weight': 1}), (8, 9, {'weight': 2}), (9, 14, {'weight': 1}), (10, 14, {'weight': 1}), (11, 12, {'weight': 1}), (12, 15, {'weight': 1}), (13, 15, {'weight': 1})]]]
    for method in self.methods:
        for G, term_nodes, valid_trees in zip([self.G1, self.G2, self.G3], [self.G1_term_nodes, self.G2_term_nodes, self.G3_term_nodes], valid_steiner_trees):
            S = steiner_tree(G, term_nodes, method=method)
            assert any((edges_equal(list(S.edges(data=True)), valid_tree) for valid_tree in valid_trees))