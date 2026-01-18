from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, tocsv, appendcsv
def test_tocsv():
    tbl = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    fn = NamedTemporaryFile().name
    tocsv(tbl, fn, encoding='utf-8', lineterminator='\n')
    expect = u'name,id\nԱրամ Խաչատրյան,1\nJohann Strauß,2\nВагиф Сәмәдоғлу,3\n章子怡,4\n'
    uf = io.open(fn, encoding='utf-8', mode='rt', newline='')
    actual = uf.read()
    eq_(expect, actual)
    tbl = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    tocsv(tbl, fn, encoding='utf-8', lineterminator='\n', write_header=False)
    expect = u'Արամ Խաչատրյան,1\nJohann Strauß,2\nВагиф Сәмәдоғлу,3\n章子怡,4\n'
    uf = io.open(fn, encoding='utf-8', mode='rt', newline='')
    actual = uf.read()
    eq_(expect, actual)