import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestMultiReverseView:

    def setup_method(self):
        self.G = nx.path_graph(9, create_using=nx.MultiDiGraph())
        self.G.add_edge(4, 5)
        self.rv = nx.reverse_view(self.G)

    def test_pickle(self):
        import pickle
        rv = self.rv
        prv = pickle.loads(pickle.dumps(rv, -1))
        assert rv._node == prv._node
        assert rv._adj == prv._adj
        assert rv.graph == prv.graph

    def test_contains(self):
        assert (2, 3, 0) in self.G.edges
        assert (3, 2, 0) not in self.G.edges
        assert (2, 3, 0) not in self.rv.edges
        assert (3, 2, 0) in self.rv.edges
        assert (5, 4, 1) in self.rv.edges
        assert (4, 5, 1) not in self.rv.edges

    def test_iter(self):
        expected = sorted(((v, u, k) for u, v, k in self.G.edges))
        assert sorted(self.rv.edges) == expected

    def test_exceptions(self):
        MG = nx.MultiGraph(self.G)
        pytest.raises(nx.NetworkXNotImplemented, nx.reverse_view, MG)