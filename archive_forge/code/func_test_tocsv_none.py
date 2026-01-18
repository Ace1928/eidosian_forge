from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, tocsv, appendcsv
def test_tocsv_none():
    tbl = ((u'col1', u'colNone'), (u'a', 1), (u'b', None), (u'c', None), (u'd', 4))
    fn = NamedTemporaryFile().name
    tocsv(tbl, fn, encoding='utf-8', lineterminator='\n')
    expect = u'col1,colNone\na,1\nb,\nc,\nd,4\n'
    uf = io.open(fn, encoding='utf-8', mode='rt', newline='')
    actual = uf.read()
    eq_(expect, actual)