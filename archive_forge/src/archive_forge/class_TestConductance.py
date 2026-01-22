import networkx as nx
class TestConductance:
    """Unit tests for the :func:`~networkx.conductance` function."""

    def test_graph(self):
        G = nx.barbell_graph(5, 0)
        S = {4}
        T = {5}
        conductance = nx.conductance(G, S, T)
        expected = 1 / 5
        assert expected == conductance
        G2 = nx.barbell_graph(3, 0)
        S2 = {0, 1, 2}
        assert nx.conductance(G2, S2) == 1 / 7