from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import logging
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, fromtsv, tocsv, appendcsv, totsv, appendtsv
def test_tocsv_appendcsv_gz():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False)
    fn = f.name + '.gz'
    f.close()
    tocsv(table, fn, encoding='ascii', lineterminator='\n')
    o = gzip.open(fn, 'rb')
    try:
        data = [b'foo,bar', b'a,1', b'b,2', b'c,2']
        expect = b'\n'.join(data) + b'\n'
        actual = o.read()
        eq_(expect, actual)
    finally:
        o.close()
    table2 = (('foo', 'bar'), ('d', 7), ('e', 9), ('f', 1))
    appendcsv(table2, fn, encoding='ascii', lineterminator='\n')
    o = gzip.open(fn, 'rb')
    try:
        data = [b'foo,bar', b'a,1', b'b,2', b'c,2', b'd,7', b'e,9', b'f,1']
        expect = b'\n'.join(data) + b'\n'
        actual = o.read()
        eq_(expect, actual)
    finally:
        o.close()