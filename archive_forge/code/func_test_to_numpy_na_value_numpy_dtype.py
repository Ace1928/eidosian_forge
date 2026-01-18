import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('values, dtype, na_value, expected', [([1, 2, None], 'float64', 0, [1.0, 2.0, 0.0]), ([Timestamp('2000'), Timestamp('2000'), pd.NaT], None, Timestamp('2000'), [np.datetime64('2000-01-01T00:00:00.000000000')] * 3)])
def test_to_numpy_na_value_numpy_dtype(index_or_series, values, dtype, na_value, expected):
    obj = index_or_series(values)
    result = obj.to_numpy(dtype=dtype, na_value=na_value)
    expected = np.array(expected)
    tm.assert_numpy_array_equal(result, expected)