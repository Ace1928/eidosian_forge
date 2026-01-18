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
def test_pi_cmp_period(self):
    idx = period_range('2007-01', periods=20, freq='M')
    per = idx[10]
    result = idx < per
    exp = idx.values < idx.values[10]
    tm.assert_numpy_array_equal(result, exp)
    result = idx.values.reshape(10, 2) < per
    tm.assert_numpy_array_equal(result, exp.reshape(10, 2))
    result = idx < np.array(per)
    tm.assert_numpy_array_equal(result, exp)