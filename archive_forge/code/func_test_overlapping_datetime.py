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
def test_overlapping_datetime(self):
    s1 = Series([1, 2, 3], index=[datetime(1995, 12, 31), datetime(2000, 12, 31), datetime(2005, 12, 31)])
    s2 = Series([1, 2, 3], index=[datetime(1997, 12, 31), datetime(2003, 12, 31), datetime(2008, 12, 31)])
    _, ax = mpl.pyplot.subplots()
    s1.plot(ax=ax)
    s2.plot(ax=ax)
    s1.plot(ax=ax)