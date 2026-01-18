import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('window', range(7))
def test_rolling_corr_with_zero_variance(window):
    s = Series(np.zeros(20))
    other = Series(np.arange(20))
    assert s.rolling(window=window).corr(other=other).isna().all()