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
@pytest.mark.xfail(reason='Api changed in 3.6.0')
def test_uhf(self):
    import pandas.plotting._matplotlib.converter as conv
    idx = date_range('2012-6-22 21:59:51.960928', freq='ms', periods=500)
    df = DataFrame(np.random.default_rng(2).standard_normal((len(idx), 2)), index=idx)
    _, ax = mpl.pyplot.subplots()
    df.plot(ax=ax)
    axis = ax.get_xaxis()
    tlocs = axis.get_ticklocs()
    tlabels = axis.get_ticklabels()
    for loc, label in zip(tlocs, tlabels):
        xp = conv._from_ordinal(loc).strftime('%H:%M:%S.%f')
        rs = str(label.get_text())
        if len(rs):
            assert xp == rs