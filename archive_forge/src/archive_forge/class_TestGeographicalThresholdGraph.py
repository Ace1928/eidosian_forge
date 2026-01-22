import math
import random
from itertools import combinations
import pytest
import networkx as nx
class TestGeographicalThresholdGraph:
    """Unit tests for :func:`~networkx.geographical_threshold_graph`"""

    def test_number_of_nodes(self):
        G = nx.geographical_threshold_graph(50, 100, seed=42)
        assert len(G) == 50
        G = nx.geographical_threshold_graph(range(50), 100, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if their
        distances meet the given threshold.
        """
        G = nx.geographical_threshold_graph(50, 10)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert join(G, u, v, 10, -2, math.dist)
            else:
                assert not join(G, u, v, 10, -2, math.dist)

    def test_metric(self):
        """Tests for providing an alternate distance metric to the generator."""
        G = nx.geographical_threshold_graph(50, 10, metric=l1dist)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert join(G, u, v, 10, -2, l1dist)
            else:
                assert not join(G, u, v, 10, -2, l1dist)

    def test_p_dist_zero(self):
        """Tests if p_dict = 0 returns disconnected graph with 0 edges"""

        def p_dist(dist):
            return 0
        G = nx.geographical_threshold_graph(50, 1, p_dist=p_dist)
        assert len(G.edges) == 0

    def test_pos_weight_name(self):
        gtg = nx.geographical_threshold_graph
        G = gtg(50, 100, seed=42, pos_name='coords', weight_name='wt')
        assert all((len(d['coords']) == 2 for n, d in G.nodes.items()))
        assert all((d['wt'] > 0 for n, d in G.nodes.items()))