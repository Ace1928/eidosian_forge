from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_get_field():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    table = pa.Table.from_arrays(data, names=('a', 'b', 'c'))
    assert table.field('a').equals(table.schema.field('a'))
    assert table.field(0).equals(table.schema.field('a'))
    with pytest.raises(KeyError):
        table.field('d')
    with pytest.raises(TypeError):
        table.field(None)
    with pytest.raises(IndexError):
        table.field(4)