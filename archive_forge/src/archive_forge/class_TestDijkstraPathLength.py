import pytest
import networkx as nx
from networkx.utils import pairwise
class TestDijkstraPathLength:
    """Unit tests for the :func:`networkx.dijkstra_path_length`
    function.

    """

    def test_weight_function(self):
        """Tests for computing the length of the shortest path using
        Dijkstra's algorithm with a user-defined weight function.

        """
        G = nx.complete_graph(3)
        G.adj[0][2]['weight'] = 10
        G.adj[0][1]['weight'] = 1
        G.adj[1][2]['weight'] = 1

        def weight(u, v, d):
            return 1 / d['weight']
        length = nx.dijkstra_path_length(G, 0, 2, weight=weight)
        assert length == 1 / 10