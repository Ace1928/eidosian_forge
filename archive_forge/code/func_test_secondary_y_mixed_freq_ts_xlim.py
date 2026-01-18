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
def test_secondary_y_mixed_freq_ts_xlim(self):
    rng = date_range('2000-01-01', periods=10000, freq='min')
    ts = Series(1, index=rng)
    _, ax = mpl.pyplot.subplots()
    ts.plot(ax=ax)
    left_before, right_before = ax.get_xlim()
    ts.resample('D').mean().plot(secondary_y=True, ax=ax)
    left_after, right_after = ax.get_xlim()
    assert left_before == left_after
    assert right_before == right_after