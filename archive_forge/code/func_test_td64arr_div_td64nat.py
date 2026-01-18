from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_div_td64nat(self, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    rng = timedelta_range('1 days', '10 days')
    rng = tm.box_expected(rng, box)
    other = np.timedelta64('NaT')
    expected = np.array([np.nan] * 10)
    expected = tm.box_expected(expected, xbox)
    result = rng / other
    tm.assert_equal(result, expected)
    result = other / rng
    tm.assert_equal(result, expected)