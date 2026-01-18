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
def test_pi_add_sub_int_array_freqn_gt1(self):
    pi = period_range('2016-01-01', periods=10, freq='2D')
    arr = np.arange(10)
    result = pi + arr
    expected = pd.Index([x + y for x, y in zip(pi, arr)])
    tm.assert_index_equal(result, expected)
    result = pi - arr
    expected = pd.Index([x - y for x, y in zip(pi, arr)])
    tm.assert_index_equal(result, expected)