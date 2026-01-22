import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestEdgeView:

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.eview = nx.reportviews.EdgeView

    def test_pickle(self):
        import pickle
        ev = self.eview(self.G)
        pev = pickle.loads(pickle.dumps(ev, -1))
        assert ev == pev
        assert ev.__slots__ == pev.__slots__

    def modify_edge(self, G, e, **kwds):
        G._adj[e[0]][e[1]].update(kwds)

    def test_str(self):
        ev = self.eview(self.G)
        rep = str([(n, n + 1) for n in range(8)])
        assert str(ev) == rep

    def test_repr(self):
        ev = self.eview(self.G)
        rep = 'EdgeView([(0, 1), (1, 2), (2, 3), (3, 4), ' + '(4, 5), (5, 6), (6, 7), (7, 8)])'
        assert repr(ev) == rep

    def test_getitem(self):
        G = self.G.copy()
        ev = G.edges
        G.edges[0, 1]['foo'] = 'bar'
        assert ev[0, 1] == {'foo': 'bar'}
        with pytest.raises(nx.NetworkXError):
            G.edges[0:5]

    def test_call(self):
        ev = self.eview(self.G)
        assert id(ev) == id(ev())
        assert id(ev) == id(ev(data=False))
        assert id(ev) != id(ev(data=True))
        assert id(ev) != id(ev(nbunch=1))

    def test_data(self):
        ev = self.eview(self.G)
        assert id(ev) != id(ev.data())
        assert id(ev) == id(ev.data(data=False))
        assert id(ev) != id(ev.data(data=True))
        assert id(ev) != id(ev.data(nbunch=1))

    def test_iter(self):
        ev = self.eview(self.G)
        for u, v in ev:
            pass
        iev = iter(ev)
        assert next(iev) == (0, 1)
        assert iter(ev) != ev
        assert iter(iev) == iev

    def test_contains(self):
        ev = self.eview(self.G)
        edv = ev()
        if self.G.is_directed():
            assert (1, 2) in ev and (2, 1) not in ev
            assert (1, 2) in edv and (2, 1) not in edv
        else:
            assert (1, 2) in ev and (2, 1) in ev
            assert (1, 2) in edv and (2, 1) in edv
        assert (1, 4) not in ev
        assert (1, 4) not in edv
        assert (1, 90) not in ev
        assert (90, 1) not in ev
        assert (1, 90) not in edv
        assert (90, 1) not in edv

    def test_contains_with_nbunch(self):
        ev = self.eview(self.G)
        evn = ev(nbunch=[0, 2])
        assert (0, 1) in evn
        assert (1, 2) in evn
        assert (2, 3) in evn
        assert (3, 4) not in evn
        assert (4, 5) not in evn
        assert (5, 6) not in evn
        assert (7, 8) not in evn
        assert (8, 9) not in evn

    def test_len(self):
        ev = self.eview(self.G)
        num_ed = 9 if self.G.is_multigraph() else 8
        assert len(ev) == num_ed
        H = self.G.copy()
        H.add_edge(1, 1)
        assert len(H.edges(1)) == 3 + H.is_multigraph() - H.is_directed()
        assert len(H.edges()) == num_ed + 1
        assert len(H.edges) == num_ed + 1

    def test_and(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1), (1, 0), (0, 2)}
        if self.G.is_directed():
            assert some_edges & ev, {(0, 1)}
            assert ev & some_edges, {(0, 1)}
        else:
            assert ev & some_edges == {(0, 1), (1, 0)}
            assert some_edges & ev == {(0, 1), (1, 0)}
        return

    def test_or(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1), (1, 0), (0, 2)}
        result1 = {(n, n + 1) for n in range(8)}
        result1.update(some_edges)
        result2 = {(n + 1, n) for n in range(8)}
        result2.update(some_edges)
        assert ev | some_edges in (result1, result2)
        assert some_edges | ev in (result1, result2)

    def test_xor(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1), (1, 0), (0, 2)}
        if self.G.is_directed():
            result = {(n, n + 1) for n in range(1, 8)}
            result.update({(1, 0), (0, 2)})
            assert ev ^ some_edges == result
        else:
            result = {(n, n + 1) for n in range(1, 8)}
            result.update({(0, 2)})
            assert ev ^ some_edges == result
        return

    def test_sub(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1), (1, 0), (0, 2)}
        result = {(n, n + 1) for n in range(8)}
        result.remove((0, 1))
        assert ev - some_edges, result