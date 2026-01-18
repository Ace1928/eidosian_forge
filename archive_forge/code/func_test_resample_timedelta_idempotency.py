from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_timedelta_idempotency():
    index = timedelta_range('0', periods=9, freq='10ms')
    series = Series(range(9), index=index)
    result = series.resample('10ms').mean()
    expected = series.astype(float)
    tm.assert_series_equal(result, expected)