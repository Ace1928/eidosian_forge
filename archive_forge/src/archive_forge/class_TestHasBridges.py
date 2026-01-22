import pytest
import networkx as nx
class TestHasBridges:
    """Unit tests for the has bridges function."""

    def test_single_bridge(self):
        edges = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (1, 3), (1, 4), (2, 5), (5, 10), (6, 8)]
        G = nx.Graph(edges)
        assert nx.has_bridges(G)
        assert nx.has_bridges(G, root=1)

    def test_has_bridges_raises_root_not_in_G(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        with pytest.raises(nx.NodeNotFound):
            nx.has_bridges(G, root=6)

    def test_multiedge_bridge(self):
        edges = [(0, 1), (0, 2), (1, 2), (1, 2), (2, 3), (3, 4), (3, 4)]
        G = nx.MultiGraph(edges)
        assert nx.has_bridges(G)
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])
        assert not nx.has_bridges(G)

    def test_bridges_multiple_components(self):
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2])
        nx.add_path(G, [4, 5, 6])
        assert list(nx.bridges(G, root=4)) == [(4, 5), (5, 6)]