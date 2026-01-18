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
def test_pi_add_offset_n_gt1_not_divisible(self, box_with_array):
    pi = PeriodIndex(['2016-01'], freq='2M')
    expected = PeriodIndex(['2016-04'], freq='2M')
    pi = tm.box_expected(pi, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = pi + to_offset('3ME')
    tm.assert_equal(result, expected)
    result = to_offset('3ME') + pi
    tm.assert_equal(result, expected)