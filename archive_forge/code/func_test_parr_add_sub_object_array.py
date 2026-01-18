import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_parr_add_sub_object_array(self):
    pi = period_range('2000-12-31', periods=3, freq='D')
    parr = pi.array
    other = np.array([Timedelta(days=1), pd.offsets.Day(2), 3])
    with tm.assert_produces_warning(PerformanceWarning):
        result = parr + other
    expected = PeriodIndex(['2001-01-01', '2001-01-03', '2001-01-05'], freq='D')._data.astype(object)
    tm.assert_equal(result, expected)
    with tm.assert_produces_warning(PerformanceWarning):
        result = parr - other
    expected = PeriodIndex(['2000-12-30'] * 3, freq='D')._data.astype(object)
    tm.assert_equal(result, expected)