import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:Casting complex values to real discards')
def test_numeric_result_dtype(self, any_numeric_dtype):
    if is_extension_array_dtype(any_numeric_dtype):
        dtype = 'Float64'
    else:
        dtype = 'complex128' if is_complex_dtype(any_numeric_dtype) else None
    ser = Series([0, 1], dtype=any_numeric_dtype)
    if dtype == 'complex128' and np_version_gte1p25:
        with pytest.raises(TypeError, match='^a must be an array of real numbers$'):
            ser.describe()
        return
    result = ser.describe()
    expected = Series([2.0, 0.5, ser.std(), 0, 0.25, 0.5, 0.75, 1.0], index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], dtype=dtype)
    tm.assert_series_equal(result, expected)