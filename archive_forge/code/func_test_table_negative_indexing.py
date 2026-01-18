from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_negative_indexing():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array([1.0, 2.0, 3.0, 4.0, 5.0]), pa.array(['ab', 'bc', 'cd', 'de', 'ef'])]
    table = pa.Table.from_arrays(data, names=tuple('abcd'))
    assert table[-1].equals(table[3])
    assert table[-2].equals(table[2])
    assert table[-3].equals(table[1])
    assert table[-4].equals(table[0])
    with pytest.raises(IndexError):
        table[-5]
    with pytest.raises(IndexError):
        table[4]