import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
class TestSeriesAccessor:

    def test_to_dense(self):
        ser = pd.Series([0, 1, 0, 10], dtype='Sparse[int64]')
        result = ser.sparse.to_dense()
        expected = pd.Series([0, 1, 0, 10])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('attr', ['npoints', 'density', 'fill_value', 'sp_values'])
    def test_get_attributes(self, attr):
        arr = SparseArray([0, 1])
        ser = pd.Series(arr)
        result = getattr(ser.sparse, attr)
        expected = getattr(arr, attr)
        assert result == expected

    def test_from_coo(self):
        scipy_sparse = pytest.importorskip('scipy.sparse')
        row = [0, 3, 1, 0]
        col = [0, 3, 1, 2]
        data = [4, 5, 7, 9]
        sp_array = scipy_sparse.coo_matrix((data, (row, col)))
        result = pd.Series.sparse.from_coo(sp_array)
        index = pd.MultiIndex.from_arrays([np.array([0, 0, 1, 3], dtype=np.int32), np.array([0, 2, 1, 3], dtype=np.int32)])
        expected = pd.Series([4, 9, 7, 5], index=index, dtype='Sparse[int]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('sort_labels, expected_rows, expected_cols, expected_values_pos', [(False, [('b', 2), ('a', 2), ('b', 1), ('a', 1)], [('z', 1), ('z', 2), ('x', 2), ('z', 0)], {1: (1, 0), 3: (3, 3)}), (True, [('a', 1), ('a', 2), ('b', 1), ('b', 2)], [('x', 2), ('z', 0), ('z', 1), ('z', 2)], {1: (1, 2), 3: (0, 1)})])
    def test_to_coo(self, sort_labels, expected_rows, expected_cols, expected_values_pos):
        sp_sparse = pytest.importorskip('scipy.sparse')
        values = SparseArray([0, np.nan, 1, 0, None, 3], fill_value=0)
        index = pd.MultiIndex.from_tuples([('b', 2, 'z', 1), ('a', 2, 'z', 2), ('a', 2, 'z', 1), ('a', 2, 'x', 2), ('b', 1, 'z', 1), ('a', 1, 'z', 0)])
        ss = pd.Series(values, index=index)
        expected_A = np.zeros((4, 4))
        for value, (row, col) in expected_values_pos.items():
            expected_A[row, col] = value
        A, rows, cols = ss.sparse.to_coo(row_levels=(0, 1), column_levels=(2, 3), sort_labels=sort_labels)
        assert isinstance(A, sp_sparse.coo_matrix)
        tm.assert_numpy_array_equal(A.toarray(), expected_A)
        assert rows == expected_rows
        assert cols == expected_cols

    def test_non_sparse_raises(self):
        ser = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError, match='.sparse'):
            ser.sparse.density