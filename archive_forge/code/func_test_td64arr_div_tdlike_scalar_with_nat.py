from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
    box = box_with_array
    xbox = np.ndarray if box is pd.array else box
    rng = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
    expected = Index([12, np.nan, 24], dtype=np.float64, name='foo')
    rng = tm.box_expected(rng, box)
    expected = tm.box_expected(expected, xbox)
    result = rng / two_hours
    tm.assert_equal(result, expected)
    result = two_hours / rng
    expected = 1 / expected
    tm.assert_equal(result, expected)