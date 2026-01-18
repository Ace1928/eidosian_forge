import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_parr_add_sub_timedeltalike_freq_mismatch_daily(self, not_daily, box_with_array):
    other = not_daily
    rng = period_range('2014-05-01', '2014-05-15', freq='D')
    rng = tm.box_expected(rng, box_with_array)
    msg = '|'.join(['Input has different freq(=.+)? from Period.*?\\(freq=D\\)', "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."])
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng + other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng += other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng - other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng -= other