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
def test_to_numpy_unsupported_types():
    bool_arr = pa.array([True, False, True])
    with pytest.raises(ValueError):
        bool_arr.to_numpy()
    result = bool_arr.to_numpy(zero_copy_only=False)
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(result, expected)
    null_arr = pa.array([None, None, None])
    with pytest.raises(ValueError):
        null_arr.to_numpy()
    result = null_arr.to_numpy(zero_copy_only=False)
    expected = np.array([None, None, None], dtype=object)
    np.testing.assert_array_equal(result, expected)
    arr = pa.array([1, 2, None])
    with pytest.raises(ValueError, match='with 1 nulls'):
        arr.to_numpy()