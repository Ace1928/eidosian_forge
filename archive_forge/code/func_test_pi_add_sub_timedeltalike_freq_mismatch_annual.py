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
def test_pi_add_sub_timedeltalike_freq_mismatch_annual(self, mismatched_freq):
    other = mismatched_freq
    rng = period_range('2014', '2024', freq='Y')
    msg = 'Input has different freq(=.+)? from Period.*?\\(freq=Y-DEC\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng + other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng += other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng - other
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng -= other