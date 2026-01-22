import networkx as nx
class TestNormalizedCutSize:
    """Unit tests for the :func:`~networkx.normalized_cut_size` function."""

    def test_graph(self):
        G = nx.path_graph(4)
        S = {1, 2}
        T = set(G) - S
        size = nx.normalized_cut_size(G, S, T)
        expected = 2 * (1 / 4 + 1 / 2)
        assert expected == size
        assert expected == nx.normalized_cut_size(G, S)

    def test_directed(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
        S = {1, 2}
        T = set(G) - S
        size = nx.normalized_cut_size(G, S, T)
        expected = 2 * (1 / 2 + 1 / 1)
        assert expected == size
        assert expected == nx.normalized_cut_size(G, S)