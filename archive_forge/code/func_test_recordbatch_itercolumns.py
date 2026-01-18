from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_itercolumns():
    data = [pa.array(range(5), type='int16'), pa.array([-10, -5, 0, None, 10], type='int32')]
    batch = pa.record_batch(data, ['c0', 'c1'])
    columns = []
    for col in batch.itercolumns():
        columns.append(col)
    assert batch.columns == columns
    assert batch == pa.record_batch(columns, names=batch.column_names)
    assert batch != pa.record_batch(columns[1:], names=batch.column_names[1:])
    assert batch != columns