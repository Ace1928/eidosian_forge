import networkx as nx
class TestBeamSearch:
    """Unit tests for the beam search function."""

    def test_narrow(self):
        """Tests that a narrow beam width may cause an incomplete search."""
        G = nx.cycle_graph(4)
        edges = nx.bfs_beam_edges(G, 0, identity, width=1)
        assert list(edges) == [(0, 3), (3, 2)]

    def test_wide(self):
        G = nx.cycle_graph(4)
        edges = nx.bfs_beam_edges(G, 0, identity, width=2)
        assert list(edges) == [(0, 3), (0, 1), (3, 2)]

    def test_width_none(self):
        G = nx.cycle_graph(4)
        edges = nx.bfs_beam_edges(G, 0, identity, width=None)
        assert list(edges) == [(0, 3), (0, 1), (3, 2)]