from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_tdi_mul_float_series(self, box_with_array):
    box = box_with_array
    xbox = Series if box in [Index, tm.to_array, pd.array] else box
    idx = TimedeltaIndex(np.arange(5, dtype='int64'))
    idx = tm.box_expected(idx, box)
    rng5f = np.arange(5, dtype='float64')
    expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
    expected = tm.box_expected(expected, xbox)
    result = idx * Series(rng5f + 1.0)
    tm.assert_equal(result, expected)