import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestToUndirected:

    def setup_method(self):
        self.DG = nx.path_graph(9, create_using=nx.DiGraph())
        self.uv = nx.to_undirected(self.DG)
        self.MDG = nx.path_graph(9, create_using=nx.MultiDiGraph())
        self.Muv = nx.to_undirected(self.MDG)

    def test_directed(self):
        assert self.DG.is_directed()
        assert not self.uv.is_directed()

    def test_already_directed(self):
        uu = nx.to_undirected(self.uv)
        Muu = nx.to_undirected(self.Muv)
        assert edges_equal(uu.edges, self.uv.edges)
        assert edges_equal(Muu.edges, self.Muv.edges)

    def test_pickle(self):
        import pickle
        uv = self.uv
        puv = pickle.loads(pickle.dumps(uv, -1))
        assert uv._node == puv._node
        assert uv._adj == puv._adj
        assert uv.graph == puv.graph
        assert hasattr(uv, '_graph')

    def test_contains(self):
        assert (2, 3) in self.DG.edges
        assert (3, 2) not in self.DG.edges
        assert (2, 3) in self.uv.edges
        assert (3, 2) in self.uv.edges

    def test_iter(self):
        expected = sorted(self.DG.edges)
        assert sorted(self.uv.edges) == expected