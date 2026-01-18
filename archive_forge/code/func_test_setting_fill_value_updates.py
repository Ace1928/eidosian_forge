import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_setting_fill_value_updates():
    arr = SparseArray([0.0, np.nan], fill_value=0)
    arr.fill_value = np.nan
    expected = SparseArray._simple_new(sparse_array=np.array([np.nan]), sparse_index=IntIndex(2, [1]), dtype=SparseDtype(float, np.nan))
    tm.assert_sp_array_equal(arr, expected)