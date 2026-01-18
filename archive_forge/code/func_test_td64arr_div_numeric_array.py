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
def test_td64arr_div_numeric_array(self, box_with_array, vector, any_real_numpy_dtype):
    tdser = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
    vector = vector.astype(any_real_numpy_dtype)
    expected = Series(['2.95D', '1D 23h 12m', 'NaT'], dtype='timedelta64[ns]')
    tdser = tm.box_expected(tdser, box_with_array)
    xbox = get_upcast_box(tdser, vector)
    expected = tm.box_expected(expected, xbox)
    result = tdser / vector
    tm.assert_equal(result, expected)
    pattern = '|'.join(["true_divide'? cannot use operands", 'cannot perform __div__', 'cannot perform __truediv__', 'unsupported operand', 'Cannot divide', "ufunc 'divide' cannot use operands with types"])
    with pytest.raises(TypeError, match=pattern):
        vector / tdser
    result = tdser / vector.astype(object)
    if box_with_array is DataFrame:
        expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
        expected = tm.box_expected(expected, xbox).astype(object)
        msg = "The 'downcast' keyword in fillna"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected[2] = expected[2].fillna(np.timedelta64('NaT', 'ns'), downcast=False)
    else:
        expected = [tdser[n] / vector[n] for n in range(len(tdser))]
        expected = [x if x is not NaT else np.timedelta64('NaT', 'ns') for x in expected]
        if xbox is tm.to_array:
            expected = tm.to_array(expected).astype(object)
        else:
            expected = xbox(expected, dtype=object)
    tm.assert_equal(result, expected)
    with pytest.raises(TypeError, match=pattern):
        vector.astype(object) / tdser