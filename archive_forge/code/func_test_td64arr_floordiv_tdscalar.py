from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    td = Timedelta('5m3s')
    td1 = Series([td, td, NaT], dtype='m8[ns]')
    td1 = tm.box_expected(td1, box, transpose=False)
    expected = Series([0, 0, np.nan])
    expected = tm.box_expected(expected, xbox, transpose=False)
    result = td1 // scalar_td
    tm.assert_equal(result, expected)
    expected = Series([2, 2, np.nan])
    expected = tm.box_expected(expected, xbox, transpose=False)
    result = scalar_td // td1
    tm.assert_equal(result, expected)
    result = td1.__rfloordiv__(scalar_td)
    tm.assert_equal(result, expected)