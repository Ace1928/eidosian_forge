import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('floats', [[1.1, 2.1], np.array([1.1, 2.1])])
def test_period_index_simple_new_disallows_floats(self, floats):
    with pytest.raises(AssertionError, match='<class '):
        PeriodIndex._simple_new(floats)