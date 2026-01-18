from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_type_schema_pickling(pickle_module):
    cases = [pa.int8(), pa.string(), pa.binary(), pa.binary(10), pa.list_(pa.string()), pa.map_(pa.string(), pa.int8()), pa.struct([pa.field('a', 'int8'), pa.field('b', 'string')]), pa.union([pa.field('a', pa.int8()), pa.field('b', pa.int16())], pa.lib.UnionMode_SPARSE), pa.union([pa.field('a', pa.int8()), pa.field('b', pa.int16())], pa.lib.UnionMode_DENSE), pa.time32('s'), pa.time64('us'), pa.date32(), pa.date64(), pa.timestamp('ms'), pa.timestamp('ns'), pa.decimal128(12, 2), pa.decimal256(76, 38), pa.field('a', 'string', metadata={b'foo': b'bar'}), pa.list_(pa.field('element', pa.int64())), pa.large_list(pa.field('element', pa.int64())), pa.map_(pa.field('key', pa.string(), nullable=False), pa.field('value', pa.int8()))]
    for val in cases:
        roundtripped = pickle_module.loads(pickle_module.dumps(val))
        assert val == roundtripped
    fields = []
    for i, f in enumerate(cases):
        if isinstance(f, pa.Field):
            fields.append(f)
        else:
            fields.append(pa.field('_f{}'.format(i), f))
    schema = pa.schema(fields, metadata={b'foo': b'bar'})
    roundtripped = pickle_module.loads(pickle_module.dumps(schema))
    assert schema == roundtripped