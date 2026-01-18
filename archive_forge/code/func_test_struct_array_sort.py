from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_struct_array_sort():
    arr = pa.StructArray.from_arrays([pa.array([5, 7, 7, 35], type=pa.int64()), pa.array(['foo', 'car', 'bar', 'foobar'])], names=['a', 'b'])
    sorted_arr = arr.sort('descending', by='a')
    assert sorted_arr.to_pylist() == [{'a': 35, 'b': 'foobar'}, {'a': 7, 'b': 'car'}, {'a': 7, 'b': 'bar'}, {'a': 5, 'b': 'foo'}]
    arr_with_nulls = pa.StructArray.from_arrays([pa.array([5, 7, 7, 35], type=pa.int64()), pa.array(['foo', 'car', 'bar', 'foobar'])], names=['a', 'b'], mask=pa.array([False, False, True, False]))
    sorted_arr = arr_with_nulls.sort('descending', by='a', null_placement='at_start')
    assert sorted_arr.to_pylist() == [None, {'a': 35, 'b': 'foobar'}, {'a': 7, 'b': 'car'}, {'a': 5, 'b': 'foo'}]
    sorted_arr = arr_with_nulls.sort('descending', by='a', null_placement='at_end')
    assert sorted_arr.to_pylist() == [{'a': 35, 'b': 'foobar'}, {'a': 7, 'b': 'car'}, {'a': 5, 'b': 'foo'}, None]