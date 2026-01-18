from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.tests.plotting.common import _check_ticks_props
from pandas.tseries.offsets import WeekOfMonth
def test_secondary_y_irregular_ts_xlim(self):
    from pandas.plotting._matplotlib.converter import DatetimeConverter
    ts = Series(np.arange(20, dtype=np.float64), index=date_range('2020-01-01', periods=20))
    ts_irregular = ts.iloc[[1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18]]
    _, ax = mpl.pyplot.subplots()
    ts_irregular[:5].plot(ax=ax)
    ts_irregular[5:].plot(secondary_y=True, ax=ax)
    ts_irregular[:5].plot(ax=ax)
    left, right = ax.get_xlim()
    assert left <= DatetimeConverter.convert(ts_irregular.index.min(), '', ax)
    assert right >= DatetimeConverter.convert(ts_irregular.index.max(), '', ax)