from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
@pytest.mark.parametrize('dtype', [np.object_, np.complex128, np.int64, np.uint64, np.float64, np.complex64, np.int32, np.uint32, np.float32, np.int16, np.uint16, np.int8, np.uint8, np.intp])
class TestHelpFunctions:

    def test_value_count(self, dtype, writable):
        N = 43
        expected = (np.arange(N) + N).astype(dtype)
        values = np.repeat(expected, 5)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(np.sort(keys), expected)
        assert np.all(counts == 5)

    def test_value_count_mask(self, dtype):
        if dtype == np.object_:
            pytest.skip('mask not implemented for object dtype')
        values = np.array([1] * 5, dtype=dtype)
        mask = np.zeros((5,), dtype=np.bool_)
        mask[1] = True
        mask[4] = True
        keys, counts, na_counter = ht.value_count(values, False, mask=mask)
        assert len(keys) == 2
        assert na_counter == 2

    def test_value_count_stable(self, dtype, writable):
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys, counts, _ = ht.value_count(values, False)
        tm.assert_numpy_array_equal(keys, values)
        assert np.all(counts == 1)

    def test_duplicated_first(self, dtype, writable):
        N = 100
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values.flags.writeable = writable
        result = ht.duplicated(values)
        expected = np.ones_like(values, dtype=np.bool_)
        expected[::5] = False
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_yes(self, dtype, writable):
        N = 127
        arr = np.arange(N).astype(dtype)
        values = np.arange(N).astype(dtype)
        arr.flags.writeable = writable
        values.flags.writeable = writable
        result = ht.ismember(arr, values)
        expected = np.ones_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_ismember_no(self, dtype):
        N = 17
        arr = np.arange(N).astype(dtype)
        values = (np.arange(N) + N).astype(dtype)
        result = ht.ismember(arr, values)
        expected = np.zeros_like(values, dtype=np.bool_)
        tm.assert_numpy_array_equal(result, expected)

    def test_mode(self, dtype, writable):
        if dtype in (np.int8, np.uint8):
            N = 53
        else:
            N = 11111
        values = np.repeat(np.arange(N).astype(dtype), 5)
        values[0] = 42
        values.flags.writeable = writable
        result = ht.mode(values, False)[0]
        assert result == 42

    def test_mode_stable(self, dtype, writable):
        values = np.array([2, 1, 5, 22, 3, -1, 8]).astype(dtype)
        values.flags.writeable = writable
        keys = ht.mode(values, False)[0]
        tm.assert_numpy_array_equal(keys, values)