from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_remove_column():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=('a', 'b', 'c'))
    t2 = table.remove_column(0)
    t2.validate()
    expected = pa.Table.from_arrays(data[1:], names=('b', 'c'))
    assert t2.equals(expected)