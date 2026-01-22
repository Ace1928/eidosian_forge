import networkx as nx
from networkx.algorithms.approximation import (
class TestMaxClique:
    """Unit tests for the :func:`networkx.algorithms.approximation.max_clique`
    function.

    """

    def test_null_graph(self):
        G = nx.null_graph()
        assert len(max_clique(G)) == 0

    def test_complete_graph(self):
        graph = nx.complete_graph(30)
        mc = max_clique(graph)
        assert 30 == len(mc)

    def test_maximal_by_cardinality(self):
        """Tests that the maximal clique is computed according to maximum
        cardinality of the sets.

        For more information, see pull request #1531.

        """
        G = nx.complete_graph(5)
        G.add_edge(4, 5)
        clique = max_clique(G)
        assert len(clique) > 1
        G = nx.lollipop_graph(30, 2)
        clique = max_clique(G)
        assert len(clique) > 2