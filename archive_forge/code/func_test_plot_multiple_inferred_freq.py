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
def test_plot_multiple_inferred_freq(self):
    dr = Index([datetime(2000, 1, 1), datetime(2000, 1, 6), datetime(2000, 1, 11)])
    ser = Series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    _check_plot_works(ser.plot)