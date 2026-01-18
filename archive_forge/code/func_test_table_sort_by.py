from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_sort_by():
    table = pa.table([pa.array([3, 1, 4, 2, 5]), pa.array(['b', 'a', 'b', 'a', 'c'])], names=['values', 'keys'])
    assert table.sort_by('values').to_pydict() == {'keys': ['a', 'a', 'b', 'b', 'c'], 'values': [1, 2, 3, 4, 5]}
    assert table.sort_by([('values', 'descending')]).to_pydict() == {'keys': ['c', 'b', 'b', 'a', 'a'], 'values': [5, 4, 3, 2, 1]}
    tab = pa.Table.from_arrays([pa.array([5, 7, 7, 35], type=pa.int64()), pa.array(['foo', 'car', 'bar', 'foobar'])], names=['a', 'b'])
    sorted_tab = tab.sort_by([('a', 'descending')])
    sorted_tab_dict = sorted_tab.to_pydict()
    assert sorted_tab_dict['a'] == [35, 7, 7, 5]
    assert sorted_tab_dict['b'] == ['foobar', 'car', 'bar', 'foo']
    sorted_tab = tab.sort_by([('a', 'ascending')])
    sorted_tab_dict = sorted_tab.to_pydict()
    assert sorted_tab_dict['a'] == [5, 7, 7, 35]
    assert sorted_tab_dict['b'] == ['foo', 'car', 'bar', 'foobar']