import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestEdgeDataView:

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.eview = nx.reportviews.EdgeView

    def test_pickle(self):
        import pickle
        ev = self.eview(self.G)(data=True)
        pev = pickle.loads(pickle.dumps(ev, -1))
        assert list(ev) == list(pev)
        assert ev.__slots__ == pev.__slots__

    def modify_edge(self, G, e, **kwds):
        G._adj[e[0]][e[1]].update(kwds)

    def test_str(self):
        ev = self.eview(self.G)(data=True)
        rep = str([(n, n + 1, {}) for n in range(8)])
        assert str(ev) == rep

    def test_repr(self):
        ev = self.eview(self.G)(data=True)
        rep = 'EdgeDataView([(0, 1, {}), (1, 2, {}), ' + '(2, 3, {}), (3, 4, {}), ' + '(4, 5, {}), (5, 6, {}), ' + '(6, 7, {}), (7, 8, {})])'
        assert repr(ev) == rep

    def test_iterdata(self):
        G = self.G.copy()
        evr = self.eview(G)
        ev = evr(data=True)
        ev_def = evr(data='foo', default=1)
        for u, v, d in ev:
            pass
        assert d == {}
        for u, v, wt in ev_def:
            pass
        assert wt == 1
        self.modify_edge(G, (2, 3), foo='bar')
        for e in ev:
            assert len(e) == 3
            if set(e[:2]) == {2, 3}:
                assert e[2] == {'foo': 'bar'}
                checked = True
            else:
                assert e[2] == {}
        assert checked
        for e in ev_def:
            assert len(e) == 3
            if set(e[:2]) == {2, 3}:
                assert e[2] == 'bar'
                checked_wt = True
            else:
                assert e[2] == 1
        assert checked_wt

    def test_iter(self):
        evr = self.eview(self.G)
        ev = evr()
        for u, v in ev:
            pass
        iev = iter(ev)
        assert next(iev) == (0, 1)
        assert iter(ev) != ev
        assert iter(iev) == iev

    def test_contains(self):
        evr = self.eview(self.G)
        ev = evr()
        if self.G.is_directed():
            assert (1, 2) in ev and (2, 1) not in ev
        else:
            assert (1, 2) in ev and (2, 1) in ev
        assert (1, 4) not in ev
        assert (1, 90) not in ev
        assert (90, 1) not in ev

    def test_contains_with_nbunch(self):
        evr = self.eview(self.G)
        ev = evr(nbunch=[0, 2])
        if self.G.is_directed():
            assert (0, 1) in ev
            assert (1, 2) not in ev
            assert (2, 3) in ev
        else:
            assert (0, 1) in ev
            assert (1, 2) in ev
            assert (2, 3) in ev
        assert (3, 4) not in ev
        assert (4, 5) not in ev
        assert (5, 6) not in ev
        assert (7, 8) not in ev
        assert (8, 9) not in ev

    def test_len(self):
        evr = self.eview(self.G)
        ev = evr(data='foo')
        assert len(ev) == 8
        assert len(evr(1)) == 2
        assert len(evr([1, 2, 3])) == 4
        assert len(self.G.edges(1)) == 2
        assert len(self.G.edges()) == 8
        assert len(self.G.edges) == 8
        H = self.G.copy()
        H.add_edge(1, 1)
        assert len(H.edges(1)) == 3
        assert len(H.edges()) == 9
        assert len(H.edges) == 9