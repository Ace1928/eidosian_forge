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
def test_secondary_legend_right(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(211)
    df.plot(secondary_y=['A', 'C'], mark_right=False, ax=ax)
    leg = ax.get_legend()
    assert len(leg.get_lines()) == 4
    assert leg.get_texts()[0].get_text() == 'A'
    assert leg.get_texts()[1].get_text() == 'B'
    assert leg.get_texts()[2].get_text() == 'C'
    assert leg.get_texts()[3].get_text() == 'D'
    mpl.pyplot.close(fig)