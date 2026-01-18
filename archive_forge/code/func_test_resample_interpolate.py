from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat
@all_ts
def test_resample_interpolate(frame):
    df = frame
    warn = None
    if isinstance(df.index, PeriodIndex):
        warn = FutureWarning
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        result = df.resample('1min').asfreq().interpolate()
        expected = df.resample('1min').interpolate()
    tm.assert_frame_equal(result, expected)