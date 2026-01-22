import collections
import pytest
import networkx as nx
class TestEulerize:

    def test_disconnected(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.from_edgelist([(0, 1), (2, 3)])
            nx.eulerize(G)

    def test_null_graph(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.eulerize(nx.Graph())

    def test_null_multigraph(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.eulerize(nx.MultiGraph())

    def test_on_empty_graph(self):
        with pytest.raises(nx.NetworkXError):
            nx.eulerize(nx.empty_graph(3))

    def test_on_eulerian(self):
        G = nx.cycle_graph(3)
        H = nx.eulerize(G)
        assert nx.is_isomorphic(G, H)

    def test_on_eulerian_multigraph(self):
        G = nx.MultiGraph(nx.cycle_graph(3))
        G.add_edge(0, 1)
        H = nx.eulerize(G)
        assert nx.is_eulerian(H)

    def test_on_complete_graph(self):
        G = nx.complete_graph(4)
        assert nx.is_eulerian(nx.eulerize(G))
        assert nx.is_eulerian(nx.eulerize(nx.MultiGraph(G)))

    def test_on_non_eulerian_graph(self):
        G = nx.cycle_graph(18)
        G.add_edge(0, 18)
        G.add_edge(18, 19)
        G.add_edge(17, 19)
        G.add_edge(4, 20)
        G.add_edge(20, 21)
        G.add_edge(21, 22)
        G.add_edge(22, 23)
        G.add_edge(23, 24)
        G.add_edge(24, 25)
        G.add_edge(25, 26)
        G.add_edge(26, 27)
        G.add_edge(27, 28)
        G.add_edge(28, 13)
        assert not nx.is_eulerian(G)
        G = nx.eulerize(G)
        assert nx.is_eulerian(G)
        assert nx.number_of_edges(G) == 39