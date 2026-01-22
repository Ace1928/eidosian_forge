import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
class TestOrdDict(object):

    def test_new(self):
        nl = rlc.OrdDict()
        x = (('a', 123), ('b', 456), ('c', 789))
        nl = rlc.OrdDict(x)

    def test_new_invalid(self):
        with pytest.raises(TypeError):
            rlc.OrdDict({})

    def test_notimplemented_operators(self):
        nl = rlc.OrdDict()
        nl2 = rlc.OrdDict()
        assert nl == nl
        assert nl != nl2
        with pytest.raises(TypeError):
            nl > nl2
        with pytest.raises(NotImplementedError):
            reversed(nl)
        with pytest.raises(NotImplementedError):
            nl.sort()

    def test_repr(self):
        x = (('a', 123), ('b', 456), ('c', 789))
        nl = rlc.OrdDict(x)
        assert isinstance(repr(nl), str)

    def test_iter(self):
        x = (('a', 123), ('b', 456), ('c', 789))
        nl = rlc.OrdDict(x)
        for a, b in zip(nl, x):
            assert a == b[0]

    def test_len(self):
        x = rlc.OrdDict()
        assert len(x) == 0
        x['a'] = 2
        x['b'] = 1
        assert len(x) == 2

    def test_getsetitem(self):
        x = rlc.OrdDict()
        x['a'] = 1
        assert len(x) == 1
        assert x['a'] == 1
        assert x.index('a') == 0
        x['a'] = 2
        assert len(x) == 1
        assert x['a'] == 2
        assert x.index('a') == 0
        x['b'] = 1
        assert len(x) == 2
        assert x['b'] == 1
        assert x.index('b') == 1

    def test_get(self):
        x = rlc.OrdDict()
        x['a'] = 1
        assert x.get('a') == 1
        assert x.get('b') is None
        assert x.get('b', 2) == 2

    def test_keys(self):
        x = rlc.OrdDict()
        word = 'abcdef'
        for i, k in enumerate(word):
            x[k] = i
        for i, k in enumerate(x.keys()):
            assert word[i] == k

    def test_getsetitemwithnone(self):
        x = rlc.OrdDict()
        x['a'] = 1
        x[None] = 2
        assert len(x) == 2
        x['b'] = 5
        assert len(x) == 3
        assert x['a'] == 1
        assert x['b'] == 5
        assert x.index('a') == 0
        assert x.index('b') == 2

    def test_reverse(self):
        x = rlc.OrdDict()
        x['a'] = 3
        x['b'] = 2
        x['c'] = 1
        x.reverse()
        assert x['c'] == 1
        assert x.index('c') == 0
        assert x['b'] == 2
        assert x.index('b') == 1
        assert x['a'] == 3
        assert x.index('a') == 2

    def test_items(self):
        args = (('a', 5), ('b', 4), ('c', 3), ('d', 2), ('e', 1))
        x = rlc.OrdDict(args)
        it = x.items()
        for ki, ko in zip(args, it):
            assert ki[0] == ko[0]
            assert ki[1] == ko[1]

    def test_pickling(self):
        f = BytesIO()
        pickle.dump(rlc.OrdDict([('a', 1), ('b', 2)]), f)
        f.seek(0)
        od = pickle.load(f)
        assert od['a'] == 1
        assert od.index('a') == 0
        assert od['b'] == 2
        assert od.index('b') == 1