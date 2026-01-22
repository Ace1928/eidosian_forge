import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
class TestLRUContainer(object):

    def test_maxsize(self):
        d = Container(5)
        for i in xrange(5):
            d[i] = str(i)
        assert len(d) == 5
        for i in xrange(5):
            assert d[i] == str(i)
        d[i + 1] = str(i + 1)
        assert len(d) == 5
        assert 0 not in d
        assert i + 1 in d

    def test_expire(self):
        d = Container(5)
        for i in xrange(5):
            d[i] = str(i)
        for i in xrange(5):
            d.get(0)
        d[5] = '5'
        assert list(d.keys()) == [2, 3, 4, 0, 5]

    def test_same_key(self):
        d = Container(5)
        for i in xrange(10):
            d['foo'] = i
        assert list(d.keys()) == ['foo']
        assert len(d) == 1

    def test_access_ordering(self):
        d = Container(5)
        for i in xrange(10):
            d[i] = True
        assert list(d.keys()) == [5, 6, 7, 8, 9]
        new_order = [7, 8, 6, 9, 5]
        for k in new_order:
            d[k]
        assert list(d.keys()) == new_order

    def test_delete(self):
        d = Container(5)
        for i in xrange(5):
            d[i] = True
        del d[0]
        assert 0 not in d
        d.pop(1)
        assert 1 not in d
        d.pop(1, None)

    def test_get(self):
        d = Container(5)
        for i in xrange(5):
            d[i] = True
        r = d.get(4)
        assert r is True
        r = d.get(5)
        assert r is None
        r = d.get(5, 42)
        assert r == 42
        with pytest.raises(KeyError):
            d[5]

    def test_disposal(self):
        evicted_items = []

        def dispose_func(arg):
            evicted_items.append(arg)
        d = Container(5, dispose_func=dispose_func)
        for i in xrange(5):
            d[i] = i
        assert list(d.keys()) == list(xrange(5))
        assert evicted_items == []
        d[5] = 5
        assert list(d.keys()) == list(xrange(1, 6))
        assert evicted_items == [0]
        del d[1]
        assert evicted_items == [0, 1]
        d.clear()
        assert evicted_items == [0, 1, 2, 3, 4, 5]

    def test_iter(self):
        d = Container()
        with pytest.raises(NotImplementedError):
            d.__iter__()