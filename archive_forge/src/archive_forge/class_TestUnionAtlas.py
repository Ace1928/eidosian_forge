import pickle
import pytest
import networkx as nx
class TestUnionAtlas:

    def setup_method(self):
        self.s = {0: {'color': 'blue', 'weight': 1.2}, 1: {}, 2: {'color': 1}}
        self.p = {3: {'color': 'blue', 'weight': 1.2}, 4: {}, 2: {'watch': 2}}
        self.av = nx.classes.coreviews.UnionAtlas(self.s, self.p)

    def test_pickle(self):
        view = self.av
        pview = pickle.loads(pickle.dumps(view, -1))
        assert view == pview
        assert view.__slots__ == pview.__slots__

    def test_len(self):
        assert len(self.av) == len(self.s.keys() | self.p.keys()) == 5

    def test_iter(self):
        assert set(self.av) == set(self.s) | set(self.p)

    def test_getitem(self):
        assert self.av[0] is self.s[0]
        assert self.av[4] is self.p[4]
        assert self.av[2]['color'] == 1
        pytest.raises(KeyError, self.av[2].__getitem__, 'watch')
        pytest.raises(KeyError, self.av.__getitem__, 8)

    def test_copy(self):
        avcopy = self.av.copy()
        assert avcopy[0] == self.av[0]
        assert avcopy[0] is not self.av[0]
        assert avcopy is not self.av
        avcopy[5] = {}
        assert avcopy != self.av
        avcopy[0]['ht'] = 4
        assert avcopy[0] != self.av[0]
        self.av[0]['ht'] = 4
        assert avcopy[0] == self.av[0]
        del self.av[0]['ht']
        assert not hasattr(self.av, '__setitem__')

    def test_items(self):
        expected = dict(self.p.items())
        expected.update(self.s)
        assert sorted(self.av.items()) == sorted(expected.items())

    def test_str(self):
        out = str(dict(self.av))
        assert str(self.av) == out

    def test_repr(self):
        out = f'{self.av.__class__.__name__}({self.s}, {self.p})'
        assert repr(self.av) == out