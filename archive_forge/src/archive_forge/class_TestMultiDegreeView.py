import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestMultiDegreeView(TestDegreeView):
    GRAPH = nx.MultiGraph
    dview = nx.reportviews.MultiDegreeView

    def test_str(self):
        dv = self.dview(self.G)
        rep = str([(0, 1), (1, 4), (2, 2), (3, 4), (4, 2), (5, 1)])
        assert str(dv) == rep
        dv = self.G.degree()
        assert str(dv) == rep

    def test_repr(self):
        dv = self.G.degree()
        rep = 'MultiDegreeView({0: 1, 1: 4, 2: 2, 3: 4, 4: 2, 5: 1})'
        assert repr(dv) == rep

    def test_nbunch(self):
        dv = self.dview(self.G)
        dvn = dv(0)
        assert dvn == 1
        dvn = dv([2, 3])
        assert sorted(dvn) == [(2, 2), (3, 4)]

    def test_getitem(self):
        dv = self.dview(self.G)
        assert dv[0] == 1
        assert dv[1] == 4
        assert dv[2] == 2
        assert dv[3] == 4
        dv = self.dview(self.G, weight='foo')
        assert dv[0] == 1
        assert dv[1] == 7
        assert dv[2] == 2
        assert dv[3] == 7

    def test_weight(self):
        dv = self.dview(self.G)
        dvw = dv(0, weight='foo')
        assert dvw == 1
        dvw = dv(1, weight='foo')
        assert dvw == 7
        dvw = dv([2, 3], weight='foo')
        assert sorted(dvw) == [(2, 2), (3, 7)]
        dvd = dict(dv(weight='foo'))
        assert dvd[0] == 1
        assert dvd[1] == 7
        assert dvd[2] == 2
        assert dvd[3] == 7