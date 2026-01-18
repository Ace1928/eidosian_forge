from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teetext_unicode():
    t1 = ((u'foo', u'bar'), (u'Արամ Խաչատրյան', 2), (u'Johann Strauß', 1), (u'Вагиф Сәмәдоғлу', 3))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    prologue = u'foo,bar\n'
    template = u'{foo},{bar}\n'
    epilogue = u'章子怡,4'
    etl.wrap(t1).teetext(f1.name, template=template, prologue=prologue, epilogue=epilogue, encoding='utf-8').selectgt('bar', 1).topickle(f2.name)
    ieq(t1 + ((u'章子怡', 4),), etl.fromcsv(f1.name, encoding='utf-8').convertnumbers())
    ieq(etl.wrap(t1).selectgt('bar', 1), etl.frompickle(f2.name))