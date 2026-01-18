from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_option,ascending,expected', [('top', True, [3.0, 1.0, 2.0]), ('top', False, [2.0, 1.0, 3.0]), ('bottom', True, [2.0, 3.0, 1.0]), ('bottom', False, [1.0, 3.0, 2.0])])
def test_rank_inf_nans_na_option(self, frame_or_series, method, na_option, ascending, expected):
    obj = frame_or_series([np.inf, np.nan, -np.inf])
    result = obj.rank(method=method, na_option=na_option, ascending=ascending)
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)