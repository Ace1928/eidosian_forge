import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, other, expected, dtype', [(['a', None], [None, 'b'], ['a', 'b'], 'string[python]'), pytest.param(['a', None], [None, 'b'], ['a', 'b'], 'string[pyarrow]', marks=td.skip_if_no('pyarrow')), ([1, None], [None, 2], [1, 2], 'Int64'), ([True, None], [None, False], [True, False], 'boolean'), (['a', None], [None, 'b'], ['a', 'b'], CategoricalDtype(categories=['a', 'b'])), ([Timestamp(year=2020, month=1, day=1, tz='Europe/London'), NaT], [NaT, Timestamp(year=2020, month=1, day=1, tz='Europe/London')], [Timestamp(year=2020, month=1, day=1, tz='Europe/London')] * 2, 'datetime64[ns, Europe/London]')])
def test_update_extension_array_series(self, data, other, expected, dtype):
    result = Series(data, dtype=dtype)
    other = Series(other, dtype=dtype)
    expected = Series(expected, dtype=dtype)
    result.update(other)
    tm.assert_series_equal(result, expected)