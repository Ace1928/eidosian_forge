import math
import random
from itertools import combinations
import pytest
import networkx as nx
class TestThresholdedRandomGeometricGraph:
    """Unit tests for :func:`~networkx.thresholded_random_geometric_graph`"""

    def test_number_of_nodes(self):
        G = nx.thresholded_random_geometric_graph(50, 0.2, 0.1, seed=42)
        assert len(G) == 50
        G = nx.thresholded_random_geometric_graph(range(50), 0.2, 0.1, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.
        """
        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, seed=42)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the generator."""

        def dist(x, y):
            return sum((abs(a - b) for a, b in zip(x, y)))
        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, p=1, seed=42)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string
        nodes = list(string.ascii_lowercase)
        G = nx.thresholded_random_geometric_graph(nodes, 0.25, 0.1, seed=42)
        assert len(G) == len(nodes)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_theta(self):
        """Tests that pairs of vertices adjacent if and only if their sum
        weights exceeds the threshold parameter theta.
        """
        G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, seed=42)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert G.nodes[u]['weight'] + G.nodes[v]['weight'] >= 0.1

    def test_pos_name(self):
        trgg = nx.thresholded_random_geometric_graph
        G = trgg(50, 0.25, 0.1, seed=42, pos_name='p', weight_name='wt')
        assert all((len(d['p']) == 2 for n, d in G.nodes.items()))
        assert all((d['wt'] > 0 for n, d in G.nodes.items()))