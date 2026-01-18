from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teehtml_unicode():
    t1 = ((u'foo', u'bar'), (u'Արամ Խաչատրյան', 2), (u'Johann Strauß', 1), (u'Вагиф Сәмәдоғлу', 3))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).teehtml(f1.name, encoding='utf-8').selectgt('bar', 1).topickle(f2.name)
    ieq(t1, etl.fromxml(f1.name, './/tr', ('th', 'td'), encoding='utf-8').convertnumbers())
    ieq(etl.wrap(t1).selectgt('bar', 1), etl.frompickle(f2.name))