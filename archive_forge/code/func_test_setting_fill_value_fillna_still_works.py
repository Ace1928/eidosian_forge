import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_setting_fill_value_fillna_still_works():
    arr = SparseArray([1.0, np.nan, 1.0], fill_value=0.0)
    arr.fill_value = np.nan
    result = arr.isna()
    result = np.asarray(result)
    expected = np.array([False, True, False])
    tm.assert_numpy_array_equal(result, expected)