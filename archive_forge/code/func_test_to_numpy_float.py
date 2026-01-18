import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('box', [True, False], ids=['series', 'array'])
def test_to_numpy_float(box):
    con = pd.Series if box else pd.array
    arr = con([0.1, 0.2, 0.3], dtype='Float64')
    result = arr.to_numpy(dtype='float64')
    expected = np.array([0.1, 0.2, 0.3], dtype='float64')
    tm.assert_numpy_array_equal(result, expected)
    arr = con([0.1, 0.2, None], dtype='Float64')
    result = arr.to_numpy(dtype='float64')
    expected = np.array([0.1, 0.2, np.nan], dtype='float64')
    tm.assert_numpy_array_equal(result, expected)
    result = arr.to_numpy(dtype='float64', na_value=np.nan)
    expected = np.array([0.1, 0.2, np.nan], dtype='float64')
    tm.assert_numpy_array_equal(result, expected)