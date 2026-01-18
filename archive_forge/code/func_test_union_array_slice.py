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
def test_union_array_slice():
    arr = pa.UnionArray.from_sparse(pa.array([0, 0, 1, 1], type=pa.int8()), [pa.array(['a', 'b', 'c', 'd']), pa.array([1, 2, 3, 4])])
    assert arr[1:].to_pylist() == ['b', 3, 4]
    binary = pa.array([b'a', b'b', b'c', b'd'], type='binary')
    int64 = pa.array([1, 2, 3], type='int64')
    types = pa.array([0, 1, 0, 0, 1, 1, 0], type='int8')
    value_offsets = pa.array([0, 0, 2, 1, 1, 2, 3], type='int32')
    arr = pa.UnionArray.from_dense(types, value_offsets, [binary, int64])
    lst = arr.to_pylist()
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            assert arr[i:j].to_pylist() == lst[i:j]