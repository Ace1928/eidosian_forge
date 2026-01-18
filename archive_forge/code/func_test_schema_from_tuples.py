from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_from_tuples():
    fields = [('foo', pa.int32()), ('bar', pa.string()), ('baz', pa.list_(pa.int8()))]
    sch = pa.schema(fields)
    assert sch.names == ['foo', 'bar', 'baz']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]
    assert len(sch) == 3
    assert repr(sch) == 'foo: int32\nbar: string\nbaz: list<item: int8>\n  child 0, item: int8'
    with pytest.raises(TypeError):
        pa.schema([('foo', None)])