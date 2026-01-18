from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_c_array_interface():

    class BatchWrapper:

        def __init__(self, batch):
            self.batch = batch

        def __arrow_c_array__(self, requested_schema=None):
            return self.batch.__arrow_c_array__(requested_schema)
    data = pa.record_batch([pa.array([1, 2, 3], type=pa.int64())], names=['a'])
    wrapper = BatchWrapper(data)
    result = pa.table(wrapper)
    expected = pa.Table.from_batches([data])
    assert result == expected
    castable_schema = pa.schema([pa.field('a', pa.int32())])
    result = pa.table(wrapper, schema=castable_schema)
    expected = pa.table({'a': pa.array([1, 2, 3], type=pa.int32())})
    assert result == expected