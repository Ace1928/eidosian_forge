from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_record_batch_sort():
    rb = pa.RecordBatch.from_arrays([pa.array([7, 35, 7, 5], type=pa.int64()), pa.array([4, 1, 3, 2], type=pa.int64()), pa.array(['foo', 'car', 'bar', 'foobar'])], names=['a', 'b', 'c'])
    sorted_rb = rb.sort_by([('a', 'descending'), ('b', 'descending')])
    sorted_rb_dict = sorted_rb.to_pydict()
    assert sorted_rb_dict['a'] == [35, 7, 7, 5]
    assert sorted_rb_dict['b'] == [1, 4, 3, 2]
    assert sorted_rb_dict['c'] == ['car', 'foo', 'bar', 'foobar']
    sorted_rb = rb.sort_by([('a', 'ascending'), ('b', 'ascending')])
    sorted_rb_dict = sorted_rb.to_pydict()
    assert sorted_rb_dict['a'] == [5, 7, 7, 35]
    assert sorted_rb_dict['b'] == [2, 3, 4, 1]
    assert sorted_rb_dict['c'] == ['foobar', 'bar', 'foo', 'car']