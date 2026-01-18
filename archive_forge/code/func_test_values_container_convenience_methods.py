from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import csv
from petl.compat import PY2
import petl as etl
from petl.test.helpers import ieq, eq_
def test_values_container_convenience_methods():
    table = etl.wrap((('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2)))
    actual = table.values('foo').set()
    expect = {'a', 'b', 'c'}
    eq_(expect, actual)
    actual = table.values('foo').list()
    expect = ['a', 'b', 'c']
    eq_(expect, actual)
    actual = table.values('foo').tuple()
    expect = ('a', 'b', 'c')
    eq_(expect, actual)
    actual = table.values('bar').sum()
    expect = 5
    eq_(expect, actual)
    actual = table.data().dict()
    expect = {'a': 1, 'b': 2, 'c': 2}
    eq_(expect, actual)