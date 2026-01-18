from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.compat import pickle
from petl.test.helpers import ieq
from petl.io.pickle import frompickle, topickle, appendpickle
def test_topickle_appendpickle():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False)
    topickle(table, f.name)
    with open(f.name, 'rb') as o:
        actual = picklereader(o)
        ieq(table, actual)
    table2 = (('foo', 'bar'), ('d', 7), ('e', 9), ('f', 1))
    appendpickle(table2, f.name)
    with open(f.name, 'rb') as o:
        actual = picklereader(o)
        expect = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2), ('d', 7), ('e', 9), ('f', 1))
        ieq(expect, actual)