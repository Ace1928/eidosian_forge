from __future__ import absolute_import, print_function, division
import math
from datetime import datetime, date
from decimal import Decimal
from tempfile import NamedTemporaryFile
import pytest
from petl.compat import PY3
from petl.transform.basics import cat
from petl.util.base import dicts
from petl.util.vis import look
from petl.test.helpers import ieq
from petl.io.avro import fromavro, toavro, appendavro
from petl.test.io.test_avro_schemas import schema0, schema1, schema2, \
def test_toavro_troubleshooting11():
    table0 = list(table1)
    table0[3][1] = None
    try:
        _write_temp_avro_file(table0, schema1)
    except TypeError as tex:
        joe = '%s' % tex
        assert 'Joe' in joe
        return
    assert False, 'Failed schema conversion'