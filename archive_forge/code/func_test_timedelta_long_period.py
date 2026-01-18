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
def test_timedelta_long_period(self):
    index = timedelta_range('1 day 2 hr 30 min 10 s', periods=10, freq='1 d')
    s = Series(np.random.default_rng(2).standard_normal(len(index)), index)
    _, ax = mpl.pyplot.subplots()
    _check_plot_works(s.plot, ax=ax)