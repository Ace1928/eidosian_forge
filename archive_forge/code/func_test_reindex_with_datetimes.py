import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_with_datetimes():
    rng = date_range('1/1/2000', periods=20)
    ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)
    result = ts.reindex(list(ts.index[5:10]))
    expected = ts[5:10]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)
    result = ts[list(ts.index[5:10])]
    tm.assert_series_equal(result, expected)