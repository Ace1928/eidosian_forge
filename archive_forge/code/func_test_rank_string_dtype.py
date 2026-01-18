from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype, exp_dtype', [('string[pyarrow]', 'Int64'), ('string[pyarrow_numpy]', 'float64')])
def test_rank_string_dtype(self, dtype, exp_dtype):
    pytest.importorskip('pyarrow')
    obj = Series(['foo', 'foo', None, 'foo'], dtype=dtype)
    result = obj.rank(method='first')
    expected = Series([1, 2, None, 3], dtype=exp_dtype)
    tm.assert_series_equal(result, expected)