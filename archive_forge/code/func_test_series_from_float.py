import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_series_from_float(data):
    dtype = data.dtype
    expected = pd.Series(data)
    result = pd.Series(data.to_numpy(na_value=np.nan, dtype='float'), dtype=str(dtype))
    tm.assert_series_equal(result, expected)
    expected = pd.Series(data)
    result = pd.Series(np.array(data).tolist(), dtype=str(dtype))
    tm.assert_series_equal(result, expected)