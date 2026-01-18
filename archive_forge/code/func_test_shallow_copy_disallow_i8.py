import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_shallow_copy_disallow_i8(self):
    pi = period_range('2018-01-01', periods=3, freq='2D')
    with pytest.raises(AssertionError, match='ndarray'):
        pi._shallow_copy(pi.asi8)