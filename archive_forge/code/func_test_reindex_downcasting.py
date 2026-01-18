import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_downcasting():
    s = Series(False, index=range(5))
    msg = 'Downcasting object dtype arrays on'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.shift(1).bfill()
    expected = Series(False, index=range(5))
    tm.assert_series_equal(result, expected)