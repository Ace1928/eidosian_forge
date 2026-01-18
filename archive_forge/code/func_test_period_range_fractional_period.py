import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_period_range_fractional_period(self):
    msg = "Non-integer 'periods' in pd.date_range, pd.timedelta_range"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = period_range('2007-01', periods=10.5, freq='M')
    exp = period_range('2007-01', periods=10, freq='M')
    tm.assert_index_equal(result, exp)