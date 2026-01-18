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
def test_pi_add_iadd_timedeltalike_M(self):
    rng = period_range('2014-01', '2016-12', freq='M')
    expected = period_range('2014-06', '2017-05', freq='M')
    result = rng + pd.offsets.MonthEnd(5)
    tm.assert_index_equal(result, expected)
    rng += pd.offsets.MonthEnd(5)
    tm.assert_index_equal(rng, expected)