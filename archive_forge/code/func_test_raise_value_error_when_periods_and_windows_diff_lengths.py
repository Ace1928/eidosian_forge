from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
@pytest.mark.parametrize('periods, windows', [((3, 5), 1), (7, (3, 5))])
def test_raise_value_error_when_periods_and_windows_diff_lengths(periods, windows):
    with pytest.raises(ValueError, match='Periods and windows must have same length'):
        MSTL(endog=[1, 2, 3, 4, 5], periods=periods, windows=windows)