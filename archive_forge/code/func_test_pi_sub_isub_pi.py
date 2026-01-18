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
def test_pi_sub_isub_pi(self):
    rng = period_range('1/1/2000', freq='D', periods=5)
    other = period_range('1/6/2000', freq='D', periods=5)
    off = rng.freq
    expected = pd.Index([-5 * off] * 5)
    result = rng - other
    tm.assert_index_equal(result, expected)
    rng -= other
    tm.assert_index_equal(rng, expected)