from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_on_missing_values(self):
    timedelta_NaT = np.timedelta64('NaT')
    actual = to_timedelta(Series(['00:00:01', np.nan]))
    expected = Series([np.timedelta64(1000000000, 'ns'), timedelta_NaT], dtype=f'{tm.ENDIAN}m8[ns]')
    tm.assert_series_equal(actual, expected)
    ser = Series(['00:00:01', pd.NaT], dtype='m8[ns]')
    actual = to_timedelta(ser)
    tm.assert_series_equal(actual, expected)