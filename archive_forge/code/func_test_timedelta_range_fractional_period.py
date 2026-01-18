from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_timedelta_range_fractional_period(self):
    msg = "Non-integer 'periods' in pd.date_range, pd.timedelta_range"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rng = timedelta_range('1 days', periods=10.5)
    exp = timedelta_range('1 days', periods=10)
    tm.assert_index_equal(rng, exp)