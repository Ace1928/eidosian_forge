from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teecsv_unicode_write_header():
    t1 = ((u'name', u'id'), (u'Արամ Խաչատրյան', u'1'), (u'Johann Strauß', u'2'), (u'Вагиф Сәмәдоғлу', u'3'), (u'章子怡', u'4'))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).convertnumbers().teecsv(f1.name, write_header=False, encoding='utf-8').selectgt('id', 1).tocsv(f2.name, encoding='utf-8')
    ieq(t1[1:], etl.fromcsv(f1.name, encoding='utf-8'))
    ieq(etl.wrap(t1).convertnumbers().selectgt('id', 1), etl.fromcsv(f2.name, encoding='utf-8').convertnumbers())