from __future__ import absolute_import, print_function, division
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_, assert_almost_equal
from petl.io.numpy import toarray, fromarray, torecarray
def test_toarray_dictdtype():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    a = toarray(t, dtype={'foo': 'U4'})
    assert isinstance(a, np.ndarray)
    assert isinstance(a['foo'], np.ndarray)
    assert isinstance(a['bar'], np.ndarray)
    assert isinstance(a['baz'], np.ndarray)
    eq_('appl', a['foo'][0])
    eq_('oran', a['foo'][1])
    eq_('pear', a['foo'][2])
    eq_(1, a['bar'][0])
    eq_(3, a['bar'][1])
    eq_(7, a['bar'][2])
    assert_almost_equal(2.5, a['baz'][0])
    assert_almost_equal(4.4, a['baz'][1])
    assert_almost_equal(0.1, a['baz'][2])