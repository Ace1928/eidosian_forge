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
@pytest.mark.parametrize('int_holder', [np.array, pd.Index])
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_pi_add_intarray(self, int_holder, op):
    pi = PeriodIndex([Period('2015Q1'), Period('NaT')])
    other = int_holder([4, -1])
    result = op(pi, other)
    expected = PeriodIndex([Period('2016Q1'), Period('NaT')])
    tm.assert_index_equal(result, expected)