import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('kind', ['block', 'integer'])
def test_take_filling_all_nan(self, kind):
    sparse = SparseArray([np.nan, np.nan, np.nan, np.nan, np.nan], kind=kind)
    result = sparse.take(np.array([1, 0, -1]))
    expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
    tm.assert_sp_array_equal(result, expected)
    result = sparse.take(np.array([1, 0, -1]), fill_value=True)
    expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
    tm.assert_sp_array_equal(result, expected)
    msg = "out of bounds value in 'indices'"
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, -6]))
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, 5]))
    with pytest.raises(IndexError, match=msg):
        sparse.take(np.array([1, 5]), fill_value=True)