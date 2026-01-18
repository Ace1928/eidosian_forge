from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_equals():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    metadata = {b'foo': b'bar', b'pandas': b'badger'}
    sch1 = pa.schema(fields)
    sch2 = pa.schema(fields)
    sch3 = pa.schema(fields, metadata=metadata)
    sch4 = pa.schema(fields, metadata=metadata)
    assert sch1.equals(sch2, check_metadata=True)
    assert sch3.equals(sch4, check_metadata=True)
    assert sch1.equals(sch3)
    assert not sch1.equals(sch3, check_metadata=True)
    assert not sch1.equals(sch3, check_metadata=True)
    del fields[-1]
    sch3 = pa.schema(fields)
    assert not sch1.equals(sch3)