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
def test_parr_add_sub_index(self):
    pi = period_range('2000-12-31', periods=3)
    parr = pi.array
    result = parr - pi
    expected = pi - pi
    tm.assert_index_equal(result, expected)