import math
import random
from itertools import combinations
import pytest
import networkx as nx
class TestRandomGeometricGraph:
    """Unit tests for :func:`~networkx.random_geometric_graph`"""

    def test_number_of_nodes(self):
        G = nx.random_geometric_graph(50, 0.25, seed=42)
        assert len(G) == 50
        G = nx.random_geometric_graph(range(50), 0.25, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.
        """
        G = nx.random_geometric_graph(50, 0.25)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25
            else:
                assert not math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the generator."""
        G = nx.random_geometric_graph(50, 0.25, p=1)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert l1dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25
            else:
                assert not l1dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string
        nodes = list(string.ascii_lowercase)
        G = nx.random_geometric_graph(nodes, 0.25)
        assert len(G) == len(nodes)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25
            else:
                assert not math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_pos_name(self):
        G = nx.random_geometric_graph(50, 0.25, seed=42, pos_name='coords')
        assert all((len(d['coords']) == 2 for n, d in G.nodes.items()))