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
def _write_read_from_env_matching(prefix):
    q = 0
    for variable, base_url in os.environ.items():
        if variable.upper().startswith(prefix.upper()):
            fmsg = '\n  {}: {} -> '.format(variable, base_url)
            print(fmsg, file=sys.stderr, end='')
            _write_read_into_url(base_url)
            print('DONE ', file=sys.stderr, end='')
            q += 1
    if q < 1:
        msg = "SKIPPED\n    For testing remote source define a environment variable:\n    $ export PETL_TEST_<protocol>='<protocol>://myuser:mypassword@host:port/path/to/folder'"
        print(msg, file=sys.stderr)