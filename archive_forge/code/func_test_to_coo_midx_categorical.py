import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_to_coo_midx_categorical(self):
    sp_sparse = pytest.importorskip('scipy.sparse')
    midx = pd.MultiIndex.from_arrays([pd.CategoricalIndex(list('ab'), name='x'), pd.CategoricalIndex([0, 1], name='y')])
    ser = pd.Series(1, index=midx, dtype='Sparse[int]')
    result = ser.sparse.to_coo(row_levels=['x'], column_levels=['y'])[0]
    expected = sp_sparse.coo_matrix((np.array([1, 1]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2))
    assert (result != expected).nnz == 0