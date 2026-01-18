import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_shallow_copy_empty(self):
    idx = PeriodIndex([], freq='M')
    result = idx._view()
    expected = idx
    tm.assert_index_equal(result, expected)