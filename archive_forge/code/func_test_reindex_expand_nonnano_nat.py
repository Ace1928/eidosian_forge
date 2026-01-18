import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['timedelta64', 'datetime64'])
def test_reindex_expand_nonnano_nat(dtype):
    ser = Series(np.array([1], dtype=f'{dtype}[s]'))
    result = ser.reindex(RangeIndex(2))
    expected = Series(np.array([1, getattr(np, dtype)('nat', 's')], dtype=f'{dtype}[s]'))
    tm.assert_series_equal(result, expected)