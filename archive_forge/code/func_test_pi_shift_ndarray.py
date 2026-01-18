import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pi_shift_ndarray(self):
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
    result = idx.shift(np.array([1, 2, 3, 4]))
    expected = PeriodIndex(['2011-02', '2011-04', 'NaT', '2011-08'], freq='M', name='idx')
    tm.assert_index_equal(result, expected)
    result = idx.shift(np.array([1, -2, 3, -4]))
    expected = PeriodIndex(['2011-02', '2010-12', 'NaT', '2010-12'], freq='M', name='idx')
    tm.assert_index_equal(result, expected)