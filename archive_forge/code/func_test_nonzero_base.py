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
def test_nonzero_base(self):
    idx = date_range('2012-12-20', periods=24, freq='h') + timedelta(minutes=30)
    df = DataFrame(np.arange(24), index=idx)
    _, ax = mpl.pyplot.subplots()
    df.plot(ax=ax)
    rs = ax.get_lines()[0].get_xdata()
    assert not Index(rs).is_normalized