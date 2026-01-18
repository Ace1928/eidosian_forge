from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_unsafe_casting():
    data = [pa.array(range(5), type=pa.int64()), pa.array([-10, -5, 0, 5, 10], type=pa.int32()), pa.array([1.1, 2.2, 3.3, 4.4, 5.5], type=pa.float64()), pa.array(['ab', 'bc', 'cd', 'de', 'ef'], type=pa.string())]
    table = pa.Table.from_arrays(data, names=tuple('abcd'))
    expected_data = [pa.array(range(5), type=pa.int32()), pa.array([-10, -5, 0, 5, 10], type=pa.int16()), pa.array([1, 2, 3, 4, 5], type=pa.int64()), pa.array(['ab', 'bc', 'cd', 'de', 'ef'], type=pa.string())]
    expected_table = pa.Table.from_arrays(expected_data, names=tuple('abcd'))
    target_schema = pa.schema([pa.field('a', pa.int32()), pa.field('b', pa.int16()), pa.field('c', pa.int64()), pa.field('d', pa.string())])
    with pytest.raises(pa.ArrowInvalid, match='truncated'):
        table.cast(target_schema)
    casted_table = table.cast(target_schema, safe=False)
    assert casted_table.equals(expected_table)