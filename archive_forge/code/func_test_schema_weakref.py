from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_weakref():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    schema = pa.schema(fields)
    wr = weakref.ref(schema)
    assert wr() is not None
    del schema
    assert wr() is None