from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    rng = timedelta_range('1 days', '10 days', name='foo')
    expected = Index((np.arange(10) + 1) * 12, dtype=np.float64, name='foo')
    rng = tm.box_expected(rng, box)
    expected = tm.box_expected(expected, xbox)
    result = rng / two_hours
    tm.assert_equal(result, expected)
    result = two_hours / rng
    expected = 1 / expected
    tm.assert_equal(result, expected)