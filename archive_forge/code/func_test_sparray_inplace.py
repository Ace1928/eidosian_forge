import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_sparray_inplace():
    sparray = SparseArray([0, 2, 0, 0])
    ndarray = np.array([0, 1, 2, 3])
    sparray += ndarray
    expected = SparseArray([0, 3, 2, 3], fill_value=0)
    tm.assert_sp_array_equal(sparray, expected)