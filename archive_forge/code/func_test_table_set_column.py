from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_set_column():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=('a', 'b', 'c'))
    new_field = pa.field('d', data[1].type)
    t2 = table.set_column(0, new_field, data[1])
    expected_data = list(data)
    expected_data[0] = data[1]
    expected = pa.Table.from_arrays(expected_data, names=('d', 'b', 'c'))
    assert t2.equals(expected)