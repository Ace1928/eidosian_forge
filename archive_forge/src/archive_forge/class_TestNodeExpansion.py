import networkx as nx
class TestNodeExpansion:
    """Unit tests for the :func:`~networkx.node_expansion` function."""

    def test_graph(self):
        G = nx.path_graph(8)
        S = {3, 4, 5}
        expansion = nx.node_expansion(G, S)
        expected = 5 / 3
        assert expected == expansion