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
@pytest.mark.parametrize('transpose', [True, False])
def test_pi_add_offset_n_gt1(self, box_with_array, transpose):
    per = Period('2016-01', freq='2M')
    pi = PeriodIndex([per])
    expected = PeriodIndex(['2016-03'], freq='2M')
    pi = tm.box_expected(pi, box_with_array, transpose=transpose)
    expected = tm.box_expected(expected, box_with_array, transpose=transpose)
    result = pi + per.freq
    tm.assert_equal(result, expected)
    result = per.freq + pi
    tm.assert_equal(result, expected)