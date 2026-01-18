import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('dtype', ['int64', 'float64'])
@pytest.mark.parametrize('dense_index', [True, False])
def test_series_from_coo(self, dtype, dense_index):
    sp_sparse = pytest.importorskip('scipy.sparse')
    A = sp_sparse.eye(3, format='coo', dtype=dtype)
    result = pd.Series.sparse.from_coo(A, dense_index=dense_index)
    index = pd.MultiIndex.from_tuples([np.array([0, 0], dtype=np.int32), np.array([1, 1], dtype=np.int32), np.array([2, 2], dtype=np.int32)])
    expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
    if dense_index:
        expected = expected.reindex(pd.MultiIndex.from_product(index.levels))
    tm.assert_series_equal(result, expected)