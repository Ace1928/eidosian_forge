import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestIntNumericIndex:

    @pytest.fixture(params=[np.int64, np.int32, np.int16, np.int8])
    def dtype(self, request):
        return request.param

    def test_constructor_from_list_no_dtype(self):
        index = Index([1, 2, 3])
        assert index.dtype == np.int64

    def test_constructor(self, dtype):
        index_cls = Index
        msg = f'{index_cls.__name__}\\(\\.\\.\\.\\) must be called with a collection of some kind, 5 was passed'
        with pytest.raises(TypeError, match=msg):
            index_cls(5)
        index = index_cls([-5, 0, 1, 2], dtype=dtype)
        arr = index.values.copy()
        new_index = index_cls(arr, copy=True)
        tm.assert_index_equal(new_index, index, exact=True)
        val = int(arr[0]) + 3000
        if dtype != np.int8:
            arr[0] = val
            assert new_index[0] != val
        if dtype == np.int64:
            index = index_cls([-5, 0, 1, 2], dtype=dtype)
            expected = Index([-5, 0, 1, 2], dtype=dtype)
            tm.assert_index_equal(index, expected)
            index = index_cls(iter([-5, 0, 1, 2]), dtype=dtype)
            expected = index_cls([-5, 0, 1, 2], dtype=dtype)
            tm.assert_index_equal(index, expected, exact=True)
            expected = index_cls([5, 0], dtype=dtype)
            for cls in [Index, index_cls]:
                for idx in [cls([5, 0], dtype=dtype), cls(np.array([5, 0]), dtype=dtype), cls(Series([5, 0]), dtype=dtype)]:
                    tm.assert_index_equal(idx, expected)

    def test_constructor_corner(self, dtype):
        index_cls = Index
        arr = np.array([1, 2, 3, 4], dtype=object)
        index = index_cls(arr, dtype=dtype)
        assert index.values.dtype == index.dtype
        if dtype == np.int64:
            without_dtype = Index(arr)
            assert without_dtype.dtype == object
            tm.assert_index_equal(index, without_dtype.astype(np.int64))
        arr = np.array([1, '2', 3, '4'], dtype=object)
        msg = 'Trying to coerce float values to integers'
        with pytest.raises(ValueError, match=msg):
            index_cls(arr, dtype=dtype)

    def test_constructor_coercion_signed_to_unsigned(self, any_unsigned_int_numpy_dtype):
        msg = '|'.join(['Trying to coerce negative values to unsigned integers', 'The elements provided in the data cannot all be casted'])
        with pytest.raises(OverflowError, match=msg):
            Index([-1], dtype=any_unsigned_int_numpy_dtype)

    def test_constructor_np_signed(self, any_signed_int_numpy_dtype):
        scalar = np.dtype(any_signed_int_numpy_dtype).type(1)
        result = Index([scalar])
        expected = Index([1], dtype=any_signed_int_numpy_dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_constructor_np_unsigned(self, any_unsigned_int_numpy_dtype):
        scalar = np.dtype(any_unsigned_int_numpy_dtype).type(1)
        result = Index([scalar])
        expected = Index([1], dtype=any_unsigned_int_numpy_dtype)
        tm.assert_index_equal(result, expected, exact=True)

    def test_coerce_list(self):
        arr = Index([1, 2, 3, 4])
        assert isinstance(arr, Index)
        arr = Index([1, 2, 3, 4], dtype=object)
        assert type(arr) is Index