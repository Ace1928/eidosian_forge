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
def test_toavro_troubleshooting10():
    nullable_schema = dict(schema0)
    schema_fields = nullable_schema['fields']
    for field in schema_fields:
        field['type'] = ['null', 'string']
    try:
        _write_temp_avro_file(table1, nullable_schema)
    except ValueError as vex:
        bob = '%s' % vex
        assert 'Bob' in bob
        return
    assert False, 'Failed schema conversion'