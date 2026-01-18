import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_nat_with_tz(self):
    ts = pd.Timestamp('2015/01/01', tz='UTC')
    s = pd.Series([pd.NaT, pd.Timestamp('2015/01/01', tz='UTC')])
    result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
    expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
    tm.assert_series_equal(expected, result)