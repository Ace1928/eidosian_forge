import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_shift_datetime(self):
    a = IntervalArray.from_breaks(date_range('2000', periods=4))
    result = a.shift(2)
    expected = a.take([-1, -1, 0], allow_fill=True)
    tm.assert_interval_array_equal(result, expected)
    result = a.shift(-1)
    expected = a.take([1, 2, -1], allow_fill=True)
    tm.assert_interval_array_equal(result, expected)
    msg = 'can only insert Interval objects and NA into an IntervalArray'
    with pytest.raises(TypeError, match=msg):
        a.shift(1, fill_value=np.timedelta64('NaT', 'ns'))