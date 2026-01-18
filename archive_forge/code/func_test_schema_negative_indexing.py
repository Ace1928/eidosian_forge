from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_negative_indexing():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    schema = pa.schema(fields)
    assert schema[-1].equals(schema[2])
    assert schema[-2].equals(schema[1])
    assert schema[-3].equals(schema[0])
    with pytest.raises(IndexError):
        schema[-4]
    with pytest.raises(IndexError):
        schema[3]