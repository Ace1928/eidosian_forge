from __future__ import absolute_import, print_function, division
import gzip
import bz2
import zipfile
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
import petl as etl
from petl.io.sources import MemorySource, PopenSource, ZipSource, \
def test_zipsource():
    tbl = [('foo', 'bar'), ('a', '1'), ('b', '2')]
    fn_tsv = NamedTemporaryFile().name
    etl.totsv(tbl, fn_tsv)
    fn_zip = NamedTemporaryFile().name
    z = zipfile.ZipFile(fn_zip, mode='w')
    z.write(fn_tsv, 'data.tsv')
    z.close()
    actual = etl.fromtsv(ZipSource(fn_zip, 'data.tsv'))
    ieq(tbl, actual)