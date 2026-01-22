from __future__ import absolute_import, print_function, division
import sys
import os
from importlib import import_module
import pytest
from petl.compat import PY3
from petl.test.helpers import ieq, eq_
from petl.io.avro import fromavro, toavro
from petl.io.csv import fromcsv, tocsv
from petl.io.json import fromjson, tojson
from petl.io.xlsx import fromxlsx, toxlsx
from petl.io.xls import fromxls, toxls
from petl.util.vis import look
def _build_source_url_from(base_url, filename, compression=None):
    is_local = base_url.startswith('./')
    if compression is not None:
        if is_local:
            return None
        filename = filename + '.' + compression
        import fsspec
        codec = fsspec.utils.infer_compression(filename)
        if codec is None:
            print('\n    - %s SKIPPED ' % filename, file=sys.stderr, end='')
            return None
    print('\n    - %s ' % filename, file=sys.stderr, end='')
    if is_local:
        source_url = base_url + filename
    else:
        source_url = os.path.join(base_url, filename)
    return source_url