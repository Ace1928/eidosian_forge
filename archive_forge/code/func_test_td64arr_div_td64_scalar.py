from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('m', [1, 3, 10])
@pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    ser = Series([Timedelta(days=59)] * 3)
    ser[2] = np.nan
    flat = ser
    ser = tm.box_expected(ser, box)
    expected = Series([x / np.timedelta64(m, unit) for x in flat])
    expected = tm.box_expected(expected, xbox)
    result = ser / np.timedelta64(m, unit)
    tm.assert_equal(result, expected)
    expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
    expected = tm.box_expected(expected, xbox)
    result = np.timedelta64(m, unit) / ser
    tm.assert_equal(result, expected)