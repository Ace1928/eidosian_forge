from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_stdoutsource_none(capfd):
    tbl = [('foo', 'bar'), ('a', 1), ('b', 2)]
    etl.tocsv(tbl, encoding='ascii')
    captured = capfd.readouterr()
    outp = captured.out
    if outp:
        assert outp in ('foo,bar\r\na,1\r\nb,2\r\n', 'foo,bar\na,1\nb,2\n')