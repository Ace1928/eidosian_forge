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
def test_secondary_bar_frame(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a', 'b', 'c'])
    axes = df.plot(kind='bar', secondary_y=['a', 'c'], subplots=True)
    assert axes[0].get_yaxis().get_ticks_position() == 'right'
    assert axes[1].get_yaxis().get_ticks_position() == 'left'
    assert axes[2].get_yaxis().get_ticks_position() == 'right'