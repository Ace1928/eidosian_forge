import math
import random
from itertools import combinations
import pytest
import networkx as nx
class TestWaxmanGraph:
    """Unit tests for the :func:`~networkx.waxman_graph` function."""

    def test_number_of_nodes_1(self):
        G = nx.waxman_graph(50, 0.5, 0.1, seed=42)
        assert len(G) == 50
        G = nx.waxman_graph(range(50), 0.5, 0.1, seed=42)
        assert len(G) == 50

    def test_number_of_nodes_2(self):
        G = nx.waxman_graph(50, 0.5, 0.1, L=1)
        assert len(G) == 50
        G = nx.waxman_graph(range(50), 0.5, 0.1, L=1)
        assert len(G) == 50

    def test_metric(self):
        """Tests for providing an alternate distance metric to the generator."""
        G = nx.waxman_graph(50, 0.5, 0.1, metric=l1dist)
        assert len(G) == 50

    def test_pos_name(self):
        G = nx.waxman_graph(50, 0.5, 0.1, seed=42, pos_name='coords')
        assert all((len(d['coords']) == 2 for n, d in G.nodes.items()))