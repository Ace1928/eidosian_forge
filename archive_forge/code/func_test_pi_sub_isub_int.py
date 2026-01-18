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
def test_pi_sub_isub_int(self, one):
    """
        PeriodIndex.__sub__ and __isub__ with several representations of
        the integer 1, e.g. int, np.int64, np.uint8, ...
        """
    rng = period_range('2000-01-01 09:00', freq='h', periods=10)
    result = rng - one
    expected = period_range('2000-01-01 08:00', freq='h', periods=10)
    tm.assert_index_equal(result, expected)
    rng -= one
    tm.assert_index_equal(rng, expected)