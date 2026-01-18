from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teepickle():
    t1 = (('foo', 'bar'), ('a', 2), ('b', 1), ('c', 3))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).teepickle(f1.name).selectgt('bar', 1).topickle(f2.name)
    ieq(t1, etl.frompickle(f1.name))
    ieq(etl.wrap(t1).selectgt('bar', 1), etl.frompickle(f2.name))