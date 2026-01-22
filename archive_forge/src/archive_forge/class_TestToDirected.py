import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestToDirected:

    def setup_method(self):
        self.G = nx.path_graph(9)
        self.dv = nx.to_directed(self.G)
        self.MG = nx.path_graph(9, create_using=nx.MultiGraph())
        self.Mdv = nx.to_directed(self.MG)

    def test_directed(self):
        assert not self.G.is_directed()
        assert self.dv.is_directed()

    def test_already_directed(self):
        dd = nx.to_directed(self.dv)
        Mdd = nx.to_directed(self.Mdv)
        assert edges_equal(dd.edges, self.dv.edges)
        assert edges_equal(Mdd.edges, self.Mdv.edges)

    def test_pickle(self):
        import pickle
        dv = self.dv
        pdv = pickle.loads(pickle.dumps(dv, -1))
        assert dv._node == pdv._node
        assert dv._succ == pdv._succ
        assert dv._pred == pdv._pred
        assert dv.graph == pdv.graph

    def test_contains(self):
        assert (2, 3) in self.G.edges
        assert (3, 2) in self.G.edges
        assert (2, 3) in self.dv.edges
        assert (3, 2) in self.dv.edges

    def test_iter(self):
        revd = [tuple(reversed(e)) for e in self.G.edges]
        expected = sorted(list(self.G.edges) + revd)
        assert sorted(self.dv.edges) == expected