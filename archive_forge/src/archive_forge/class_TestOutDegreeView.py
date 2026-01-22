import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestOutDegreeView(TestDegreeView):
    GRAPH = nx.DiGraph
    dview = nx.reportviews.OutDegreeView

    def test_str(self):
        dv = self.dview(self.G)
        rep = str([(0, 1), (1, 2), (2, 1), (3, 1), (4, 1), (5, 0)])
        assert str(dv) == rep
        dv = self.G.out_degree()
        assert str(dv) == rep

    def test_repr(self):
        dv = self.G.out_degree()
        rep = 'OutDegreeView({0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 0})'
        assert repr(dv) == rep

    def test_nbunch(self):
        dv = self.dview(self.G)
        dvn = dv(0)
        assert dvn == 1
        dvn = dv([2, 3])
        assert sorted(dvn) == [(2, 1), (3, 1)]

    def test_getitem(self):
        dv = self.dview(self.G)
        assert dv[0] == 1
        assert dv[1] == 2
        assert dv[2] == 1
        assert dv[3] == 1
        dv = self.dview(self.G, weight='foo')
        assert dv[0] == 1
        assert dv[1] == 4
        assert dv[2] == 1
        assert dv[3] == 1

    def test_weight(self):
        dv = self.dview(self.G)
        dvw = dv(0, weight='foo')
        assert dvw == 1
        dvw = dv(1, weight='foo')
        assert dvw == 4
        dvw = dv([2, 3], weight='foo')
        assert sorted(dvw) == [(2, 1), (3, 1)]
        dvd = dict(dv(weight='foo'))
        assert dvd[0] == 1
        assert dvd[1] == 4
        assert dvd[2] == 1
        assert dvd[3] == 1