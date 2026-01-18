from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_remove_column_empty():
    data = [pa.array(range(5))]
    table = pa.Table.from_arrays(data, names=['a'])
    t2 = table.remove_column(0)
    t2.validate()
    assert len(t2) == len(table)
    t3 = t2.add_column(0, table.field(0), table[0])
    t3.validate()
    assert t3.equals(table)