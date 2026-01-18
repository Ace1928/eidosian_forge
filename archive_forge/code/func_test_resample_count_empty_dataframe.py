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
@pytest.mark.parametrize('freq', ['ME', 'D', 'h'])
def test_resample_count_empty_dataframe(freq, empty_frame_dti):
    empty_frame_dti['a'] = []
    if freq == 'ME' and isinstance(empty_frame_dti.index, TimedeltaIndex):
        msg = "Resampling on a TimedeltaIndex requires fixed-duration `freq`, e.g. '24h' or '3D', not <MonthEnd>"
        with pytest.raises(ValueError, match=msg):
            empty_frame_dti.resample(freq)
        return
    elif freq == 'ME' and isinstance(empty_frame_dti.index, PeriodIndex):
        freq = 'M'
    warn = None
    if isinstance(empty_frame_dti.index, PeriodIndex):
        warn = FutureWarning
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        rs = empty_frame_dti.resample(freq)
    result = rs.count()
    index = _asfreq_compat(empty_frame_dti.index, freq)
    expected = DataFrame(dtype='int64', index=index, columns=Index(['a'], dtype=object))
    tm.assert_frame_equal(result, expected)