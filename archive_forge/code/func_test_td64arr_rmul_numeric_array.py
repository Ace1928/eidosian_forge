from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('vector', [np.array([20, 30, 40]), Index([20, 30, 40]), Series([20, 30, 40])], ids=lambda x: type(x).__name__)
def test_td64arr_rmul_numeric_array(self, box_with_array, vector, any_real_numpy_dtype):
    tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
    vector = vector.astype(any_real_numpy_dtype)
    expected = Series(['1180 Days', '1770 Days', 'NaT'], dtype='timedelta64[ns]')
    tdser = tm.box_expected(tdser, box_with_array)
    xbox = get_upcast_box(tdser, vector)
    expected = tm.box_expected(expected, xbox)
    result = tdser * vector
    tm.assert_equal(result, expected)
    result = vector * tdser
    tm.assert_equal(result, expected)