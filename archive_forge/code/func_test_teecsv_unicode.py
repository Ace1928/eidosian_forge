from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teecsv_unicode():
    t1 = ((u'name', u'id'), (u'Արամ Խաչատրյան', 1), (u'Johann Strauß', 2), (u'Вагиф Сәмәдоғлу', 3), (u'章子怡', 4))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).teecsv(f1.name, encoding='utf-8').selectgt('id', 1).tocsv(f2.name, encoding='utf-8')
    ieq(t1, etl.fromcsv(f1.name, encoding='utf-8').convertnumbers())
    ieq(etl.wrap(t1).selectgt('id', 1), etl.fromcsv(f2.name, encoding='utf-8').convertnumbers())