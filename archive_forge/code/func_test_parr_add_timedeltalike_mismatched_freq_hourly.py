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
def test_parr_add_timedeltalike_mismatched_freq_hourly(self, not_hourly, box_with_array):
    other = not_hourly
    rng = period_range('2014-01-01 10:00', '2014-01-05 10:00', freq='h')
    rng = tm.box_expected(rng, box_with_array)
    msg = '|'.join(['Input has different freq(=.+)? from Period.*?\\(freq=h\\)', "Cannot add/subtract timedelta-like from PeriodArray that is not an integer multiple of the PeriodArray's freq."])
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng + other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng += other