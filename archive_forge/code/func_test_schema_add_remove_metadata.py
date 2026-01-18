from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_add_remove_metadata():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    s1 = pa.schema(fields)
    assert s1.metadata is None
    metadata = {b'foo': b'bar', b'pandas': b'badger'}
    s2 = s1.with_metadata(metadata)
    assert s2.metadata == metadata
    s3 = s2.remove_metadata()
    assert s3.metadata is None
    s4 = s3.remove_metadata()
    assert s4.metadata is None