import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestDiMultiDegreeView(TestMultiDegreeView):
    GRAPH = nx.MultiDiGraph
    dview = nx.reportviews.DiMultiDegreeView

    def test_repr(self):
        dv = self.G.degree()
        rep = 'DiMultiDegreeView({0: 1, 1: 4, 2: 2, 3: 4, 4: 2, 5: 1})'
        assert repr(dv) == rep