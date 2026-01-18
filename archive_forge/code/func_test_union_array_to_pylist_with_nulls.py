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
def test_union_array_to_pylist_with_nulls():
    arr = pa.UnionArray.from_sparse(pa.array([0, 1, 0, 0, 1], type=pa.int8()), [pa.array([0.0, 1.1, None, 3.3, 4.4]), pa.array([True, None, False, True, False])])
    assert arr.to_pylist() == [0.0, None, None, 3.3, False]
    arr = pa.UnionArray.from_dense(pa.array([0, 1, 0, 0, 0, 1, 1], type=pa.int8()), pa.array([0, 0, 1, 2, 3, 1, 2], type=pa.int32()), [pa.array([0.0, 1.1, None, 3.3]), pa.array([True, None, False])])
    assert arr.to_pylist() == [0.0, True, 1.1, None, 3.3, None, False]