from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_merge():
    a = pa.schema([pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))])
    b = pa.schema([pa.field('foo', pa.int32()), pa.field('qux', pa.bool_())])
    c = pa.schema([pa.field('quux', pa.dictionary(pa.int32(), pa.string()))])
    d = pa.schema([pa.field('foo', pa.int64()), pa.field('qux', pa.bool_())])
    result = pa.unify_schemas([a, b, c])
    expected = pa.schema([pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8())), pa.field('qux', pa.bool_()), pa.field('quux', pa.dictionary(pa.int32(), pa.string()))])
    assert result.equals(expected)
    with pytest.raises(pa.ArrowTypeError):
        pa.unify_schemas([b, d])
    result = pa.unify_schemas((a, b, c))
    assert result.equals(expected)
    result = pa.unify_schemas([b, d], promote_options='permissive')
    assert result.equals(d)
    with pytest.raises(TypeError):
        pa.unify_schemas([a, 1])