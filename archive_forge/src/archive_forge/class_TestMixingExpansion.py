import networkx as nx
class TestMixingExpansion:
    """Unit tests for the :func:`~networkx.mixing_expansion` function."""

    def test_graph(self):
        G = nx.barbell_graph(5, 0)
        S = set(range(5))
        T = set(G) - S
        expansion = nx.mixing_expansion(G, S, T)
        expected = 1 / (2 * (5 * 4 + 1))
        assert expected == expansion