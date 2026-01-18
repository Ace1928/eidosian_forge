from __future__ import absolute_import, print_function, division
import pytest
import petl as etl
from petl.test.helpers import ieq, eq_, assert_almost_equal
from petl.io.numpy import toarray, fromarray, torecarray
def test_valuesarray_explicit_dtype():
    t = [('foo', 'bar', 'baz'), ('apples', 1, 2.5), ('oranges', 3, 4.4), ('pears', 7, 0.1)]
    expect = np.array([1, 3, 7], dtype='i2')
    actual = etl.wrap(t).values('bar').array(dtype='i2')
    eq_(expect.dtype, actual.dtype)
    assert np.all(expect == actual)