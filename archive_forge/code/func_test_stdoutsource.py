from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_stdoutsource():
    tbl = [('foo', 'bar'), ('a', 1), ('b', 2)]
    etl.tocsv(tbl, StdoutSource(), encoding='ascii')
    etl.tohtml(tbl, StdoutSource(), encoding='ascii')
    etl.topickle(tbl, StdoutSource())