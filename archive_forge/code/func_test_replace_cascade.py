import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('inplace', [True, False])
def test_replace_cascade(self, inplace):
    ser = pd.Series([1, 2, 3])
    expected = pd.Series([2, 3, 4])
    res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
    if inplace:
        tm.assert_series_equal(ser, expected)
    else:
        tm.assert_series_equal(res, expected)