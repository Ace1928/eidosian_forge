import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestReverseView:

    def setup_method(self):
        self.G = nx.path_graph(9, create_using=nx.DiGraph())
        self.rv = nx.reverse_view(self.G)

    def test_pickle(self):
        import pickle
        rv = self.rv
        prv = pickle.loads(pickle.dumps(rv, -1))
        assert rv._node == prv._node
        assert rv._adj == prv._adj
        assert rv.graph == prv.graph

    def test_contains(self):
        assert (2, 3) in self.G.edges
        assert (3, 2) not in self.G.edges
        assert (2, 3) not in self.rv.edges
        assert (3, 2) in self.rv.edges

    def test_iter(self):
        expected = sorted((tuple(reversed(e)) for e in self.G.edges))
        assert sorted(self.rv.edges) == expected

    def test_exceptions(self):
        G = nx.Graph()
        pytest.raises(nx.NetworkXNotImplemented, nx.reverse_view, G)

    def test_subclass(self):

        class MyGraph(nx.DiGraph):

            def my_method(self):
                return 'me'

            def to_directed_class(self):
                return MyGraph()
        M = MyGraph()
        M.add_edge(1, 2)
        RM = nx.reverse_view(M)
        print('RM class', RM.__class__)
        RMC = RM.copy()
        print('RMC class', RMC.__class__)
        print(RMC.edges)
        assert RMC.has_edge(2, 1)
        assert RMC.my_method() == 'me'