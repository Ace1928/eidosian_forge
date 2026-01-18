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
def test_parr_sub_pi_mismatched_freq(self, box_with_array, box_with_array2):
    rng = period_range('1/1/2000', freq='D', periods=5)
    other = period_range('1/6/2000', freq='h', periods=5)
    rng = tm.box_expected(rng, box_with_array)
    other = tm.box_expected(other, box_with_array2)
    msg = 'Input has different freq=[hD] from PeriodArray\\(freq=[Dh]\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        rng - other