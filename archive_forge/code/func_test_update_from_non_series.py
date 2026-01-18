import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('series, other, expected', [(Series({'a': 1, 'b': 2, 'c': 3, 'd': 4}), {'b': 5, 'c': np.nan}, Series({'a': 1, 'b': 5, 'c': 3, 'd': 4})), (Series([1, 2, 3, 4]), [np.nan, 5, 1], Series([1, 5, 1, 4]))])
def test_update_from_non_series(self, series, other, expected):
    series.update(other)
    tm.assert_series_equal(series, expected)