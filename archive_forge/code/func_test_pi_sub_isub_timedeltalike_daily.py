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
def test_pi_sub_isub_timedeltalike_daily(self, three_days):
    other = three_days
    rng = period_range('2014-05-01', '2014-05-15', freq='D')
    expected = period_range('2014-04-28', '2014-05-12', freq='D')
    result = rng - other
    tm.assert_index_equal(result, expected)
    rng -= other
    tm.assert_index_equal(rng, expected)