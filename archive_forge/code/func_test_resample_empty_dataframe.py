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
def test_resample_empty_dataframe(empty_frame_dti, freq, resample_method):
    df = empty_frame_dti
    if freq == 'ME' and isinstance(df.index, TimedeltaIndex):
        msg = "Resampling on a TimedeltaIndex requires fixed-duration `freq`, e.g. '24h' or '3D', not <MonthEnd>"
        with pytest.raises(ValueError, match=msg):
            df.resample(freq, group_keys=False)
        return
    elif freq == 'ME' and isinstance(df.index, PeriodIndex):
        freq = 'M'
    warn = None
    if isinstance(df.index, PeriodIndex):
        warn = FutureWarning
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        rs = df.resample(freq, group_keys=False)
    result = getattr(rs, resample_method)()
    if resample_method == 'ohlc':
        mi = MultiIndex.from_product([df.columns, ['open', 'high', 'low', 'close']])
        expected = DataFrame([], index=df.index[:0].copy(), columns=mi, dtype=np.float64)
        expected.index = _asfreq_compat(df.index, freq)
    elif resample_method != 'size':
        expected = df.copy()
    else:
        expected = Series([], dtype=np.int64)
    expected.index = _asfreq_compat(df.index, freq)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq
    tm.assert_almost_equal(result, expected)