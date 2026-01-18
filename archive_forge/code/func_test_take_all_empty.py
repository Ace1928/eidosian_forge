import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_take_all_empty(self):
    sparse = pd.array([0, 0], dtype=SparseDtype('int64'))
    result = sparse.take([0, 1], allow_fill=True, fill_value=np.nan)
    tm.assert_sp_array_equal(sparse, result)