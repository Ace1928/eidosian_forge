import collections
import pytest
import networkx as nx
class TestEulerianPath:

    def test_eulerian_path(self):
        x = [(4, 0), (0, 1), (1, 2), (2, 0)]
        for e1, e2 in zip(x, nx.eulerian_path(nx.DiGraph(x))):
            assert e1 == e2

    def test_eulerian_path_straight_link(self):
        G = nx.DiGraph()
        result = [(1, 2), (2, 3), (3, 4), (4, 5)]
        G.add_edges_from(result)
        assert result == list(nx.eulerian_path(G))
        assert result == list(nx.eulerian_path(G, source=1))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=3))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=4))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=5))

    def test_eulerian_path_multigraph(self):
        G = nx.MultiDiGraph()
        result = [(2, 1), (1, 2), (2, 1), (1, 2), (2, 3), (3, 4), (4, 3)]
        G.add_edges_from(result)
        assert result == list(nx.eulerian_path(G))
        assert result == list(nx.eulerian_path(G, source=2))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=3))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=4))

    def test_eulerian_path_eulerian_circuit(self):
        G = nx.DiGraph()
        result = [(1, 2), (2, 3), (3, 4), (4, 1)]
        result2 = [(2, 3), (3, 4), (4, 1), (1, 2)]
        result3 = [(3, 4), (4, 1), (1, 2), (2, 3)]
        G.add_edges_from(result)
        assert result == list(nx.eulerian_path(G))
        assert result == list(nx.eulerian_path(G, source=1))
        assert result2 == list(nx.eulerian_path(G, source=2))
        assert result3 == list(nx.eulerian_path(G, source=3))

    def test_eulerian_path_undirected(self):
        G = nx.Graph()
        result = [(1, 2), (2, 3), (3, 4), (4, 5)]
        result2 = [(5, 4), (4, 3), (3, 2), (2, 1)]
        G.add_edges_from(result)
        assert list(nx.eulerian_path(G)) in (result, result2)
        assert result == list(nx.eulerian_path(G, source=1))
        assert result2 == list(nx.eulerian_path(G, source=5))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=3))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=2))

    def test_eulerian_path_multigraph_undirected(self):
        G = nx.MultiGraph()
        result = [(2, 1), (1, 2), (2, 1), (1, 2), (2, 3), (3, 4)]
        G.add_edges_from(result)
        assert result == list(nx.eulerian_path(G))
        assert result == list(nx.eulerian_path(G, source=2))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=3))
        with pytest.raises(nx.NetworkXError):
            list(nx.eulerian_path(G, source=1))

    @pytest.mark.parametrize(('graph_type', 'result'), ((nx.MultiGraph, [(0, 1, 0), (1, 0, 1)]), (nx.MultiDiGraph, [(0, 1, 0), (1, 0, 0)])))
    def test_eulerian_with_keys(self, graph_type, result):
        G = graph_type([(0, 1), (1, 0)])
        answer = nx.eulerian_path(G, keys=True)
        assert list(answer) == result