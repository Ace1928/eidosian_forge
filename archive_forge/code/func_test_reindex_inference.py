import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_inference():
    s = Series([True, False, False, True], index=list('abcd'))
    new_index = 'agc'
    msg = 'Downcasting object dtype arrays on'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.reindex(list(new_index)).ffill()
    expected = Series([True, True, False], index=list(new_index))
    tm.assert_series_equal(result, expected)