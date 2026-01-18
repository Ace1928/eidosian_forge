from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(('method', 'values'), [('skew', [2.0, 0.854563, 0.0, 1.999984]), ('kurt', [4.0, -1.289256, -1.2, 3.999946])])
def test_rolling_skew_kurt_large_value_range(method, values):
    s = Series([3000000, 1, 1, 2, 3, 4, 999])
    result = getattr(s.rolling(4), method)()
    expected = Series([np.nan] * 3 + values)
    tm.assert_series_equal(result, expected)