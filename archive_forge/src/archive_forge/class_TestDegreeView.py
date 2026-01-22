import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestDegreeView:
    GRAPH = nx.Graph
    dview = nx.reportviews.DegreeView

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(6, cls.GRAPH())
        cls.G.add_edge(1, 3, foo=2)
        cls.G.add_edge(1, 3, foo=3)

    def test_pickle(self):
        import pickle
        deg = self.G.degree
        pdeg = pickle.loads(pickle.dumps(deg, -1))
        assert dict(deg) == dict(pdeg)

    def test_str(self):
        dv = self.dview(self.G)
        rep = str([(0, 1), (1, 3), (2, 2), (3, 3), (4, 2), (5, 1)])
        assert str(dv) == rep
        dv = self.G.degree()
        assert str(dv) == rep

    def test_repr(self):
        dv = self.dview(self.G)
        rep = 'DegreeView({0: 1, 1: 3, 2: 2, 3: 3, 4: 2, 5: 1})'
        assert repr(dv) == rep

    def test_iter(self):
        dv = self.dview(self.G)
        for n, d in dv:
            pass
        idv = iter(dv)
        assert iter(dv) != dv
        assert iter(idv) == idv
        assert next(idv) == (0, dv[0])
        assert next(idv) == (1, dv[1])
        dv = self.dview(self.G, weight='foo')
        for n, d in dv:
            pass
        idv = iter(dv)
        assert iter(dv) != dv
        assert iter(idv) == idv
        assert next(idv) == (0, dv[0])
        assert next(idv) == (1, dv[1])

    def test_nbunch(self):
        dv = self.dview(self.G)
        dvn = dv(0)
        assert dvn == 1
        dvn = dv([2, 3])
        assert sorted(dvn) == [(2, 2), (3, 3)]

    def test_getitem(self):
        dv = self.dview(self.G)
        assert dv[0] == 1
        assert dv[1] == 3
        assert dv[2] == 2
        assert dv[3] == 3
        dv = self.dview(self.G, weight='foo')
        assert dv[0] == 1
        assert dv[1] == 5
        assert dv[2] == 2
        assert dv[3] == 5

    def test_weight(self):
        dv = self.dview(self.G)
        dvw = dv(0, weight='foo')
        assert dvw == 1
        dvw = dv(1, weight='foo')
        assert dvw == 5
        dvw = dv([2, 3], weight='foo')
        assert sorted(dvw) == [(2, 2), (3, 5)]
        dvd = dict(dv(weight='foo'))
        assert dvd[0] == 1
        assert dvd[1] == 5
        assert dvd[2] == 2
        assert dvd[3] == 5

    def test_len(self):
        dv = self.dview(self.G)
        assert len(dv) == 6