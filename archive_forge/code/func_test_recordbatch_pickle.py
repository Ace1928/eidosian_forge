from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_pickle(pickle_module):
    data = [pa.array(range(5), type='int8'), pa.array([-10, -5, 0, 5, 10], type='float32')]
    fields = [pa.field('ints', pa.int8()), pa.field('floats', pa.float32())]
    schema = pa.schema(fields, metadata={b'foo': b'bar'})
    batch = pa.record_batch(data, schema=schema)
    result = pickle_module.loads(pickle_module.dumps(batch))
    assert result.equals(batch)
    assert result.schema == schema