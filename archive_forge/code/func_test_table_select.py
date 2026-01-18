from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_select():
    a1 = pa.array([1, 2, 3, None, 5])
    a2 = pa.array(['a', 'b', 'c', 'd', 'e'])
    a3 = pa.array([[1, 2], [3, 4], [5, 6], None, [9, 10]])
    table = pa.table([a1, a2, a3], ['f1', 'f2', 'f3'])
    result = table.select(['f1'])
    expected = pa.table([a1], ['f1'])
    assert result.equals(expected)
    result = table.select(['f3', 'f2'])
    expected = pa.table([a3, a2], ['f3', 'f2'])
    assert result.equals(expected)
    result = table.select([0])
    expected = pa.table([a1], ['f1'])
    assert result.equals(expected)
    result = table.select([2, 1])
    expected = pa.table([a3, a2], ['f3', 'f2'])
    assert result.equals(expected)
    table2 = table.replace_schema_metadata({'a': 'test'})
    result = table2.select(['f1', 'f2'])
    assert b'a' in result.schema.metadata
    with pytest.raises(KeyError, match='Field "f5" does not exist'):
        table.select(['f5'])
    with pytest.raises(IndexError, match='index out of bounds'):
        table.select([5])
    result = table.select(['f2', 'f2'])
    expected = pa.table([a2, a2], ['f2', 'f2'])
    assert result.equals(expected)
    table = pa.table([a1, a2, a3], ['f1', 'f2', 'f1'])
    with pytest.raises(KeyError, match='Field "f1" exists 2 times'):
        table.select(['f1'])
    result = table.select(['f2'])
    expected = pa.table([a2], ['f2'])
    assert result.equals(expected)