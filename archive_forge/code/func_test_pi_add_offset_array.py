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
@pytest.mark.parametrize('box', [np.array, pd.Index])
def test_pi_add_offset_array(self, box):
    pi = PeriodIndex([Period('2015Q1'), Period('2016Q2')])
    offs = box([pd.offsets.QuarterEnd(n=1, startingMonth=12), pd.offsets.QuarterEnd(n=-2, startingMonth=12)])
    expected = PeriodIndex([Period('2015Q2'), Period('2015Q4')]).astype(object)
    with tm.assert_produces_warning(PerformanceWarning):
        res = pi + offs
    tm.assert_index_equal(res, expected)
    with tm.assert_produces_warning(PerformanceWarning):
        res2 = offs + pi
    tm.assert_index_equal(res2, expected)
    unanchored = np.array([pd.offsets.Hour(n=1), pd.offsets.Minute(n=-2)])
    msg = 'Input cannot be converted to Period\\(freq=Q-DEC\\)'
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            pi + unanchored
    with pytest.raises(IncompatibleFrequency, match=msg):
        with tm.assert_produces_warning(PerformanceWarning):
            unanchored + pi