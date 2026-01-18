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
def test_pi_ops_nat(self):
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='M', name='idx')
    expected = PeriodIndex(['2011-03', '2011-04', 'NaT', '2011-06'], freq='M', name='idx')
    self._check(idx, lambda x: x + 2, expected)
    self._check(idx, lambda x: 2 + x, expected)
    self._check(idx, lambda x: np.add(x, 2), expected)
    self._check(idx + 2, lambda x: x - 2, idx)
    self._check(idx + 2, lambda x: np.subtract(x, 2), idx)
    idx = PeriodIndex(['2011-01', '2011-02', 'NaT', '2011-04'], freq='2M', name='idx')
    expected = PeriodIndex(['2011-07', '2011-08', 'NaT', '2011-10'], freq='2M', name='idx')
    self._check(idx, lambda x: x + 3, expected)
    self._check(idx, lambda x: 3 + x, expected)
    self._check(idx, lambda x: np.add(x, 3), expected)
    self._check(idx + 3, lambda x: x - 3, idx)
    self._check(idx + 3, lambda x: np.subtract(x, 3), idx)