from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_get_fields():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    schema = pa.schema(fields)
    assert schema.field('foo').name == 'foo'
    assert schema.field(0).name == 'foo'
    assert schema.field(-1).name == 'baz'
    with pytest.raises(KeyError):
        schema.field('other')
    with pytest.raises(TypeError):
        schema.field(0.0)
    with pytest.raises(IndexError):
        schema.field(4)