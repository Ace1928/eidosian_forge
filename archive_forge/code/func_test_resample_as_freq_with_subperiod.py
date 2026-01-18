from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_as_freq_with_subperiod():
    index = timedelta_range('00:00:00', '00:10:00', freq='5min')
    df = DataFrame(data={'value': [1, 5, 10]}, index=index)
    result = df.resample('2min').asfreq()
    expected_data = {'value': [1, np.nan, np.nan, np.nan, np.nan, 10]}
    expected = DataFrame(data=expected_data, index=timedelta_range('00:00:00', '00:10:00', freq='2min'))
    tm.assert_frame_equal(result, expected)