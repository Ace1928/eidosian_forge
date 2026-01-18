from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('ser, exp', [([1], [1.0]), ([1, 2], [1.0 / 2, 2.0 / 2]), ([2, 2], [1.5 / 2, 1.5 / 2]), ([1, 2, 3], [1.0 / 3, 2.0 / 3, 3.0 / 3]), ([1, 2, 2], [1.0 / 3, 2.5 / 3, 2.5 / 3]), ([4, 2, 1], [3.0 / 3, 2.0 / 3, 1.0 / 3]), ([1, 1, 5, 5, 3], [1.5 / 5, 1.5 / 5, 4.5 / 5, 4.5 / 5, 3.0 / 5]), ([1, 1, 3, 3, 5, 5], [1.5 / 6, 1.5 / 6, 3.5 / 6, 3.5 / 6, 5.5 / 6, 5.5 / 6]), ([-5, -4, -3, -2, -1], [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5])])
def test_rank_average_pct(dtype, ser, exp):
    s = Series(ser).astype(dtype)
    result = s.rank(method='average', pct=True)
    expected = Series(exp).astype(result.dtype)
    tm.assert_series_equal(result, expected)