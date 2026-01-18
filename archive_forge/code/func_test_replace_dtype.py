import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('dtype, input_data, to_replace, expected_data', [('bool', [True, False], {True: False}, [False, False]), ('int64', [1, 2], {1: 10, 2: 20}, [10, 20]), ('Int64', [1, 2], {1: 10, 2: 20}, [10, 20]), ('float64', [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]), ('Float64', [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]), ('string', ['one', 'two'], {'one': '1', 'two': '2'}, ['1', '2']), (pd.IntervalDtype('int64'), IntervalArray([pd.Interval(1, 2), pd.Interval(2, 3)]), {pd.Interval(1, 2): pd.Interval(10, 20)}, IntervalArray([pd.Interval(10, 20), pd.Interval(2, 3)])), (pd.IntervalDtype('float64'), IntervalArray([pd.Interval(1.0, 2.7), pd.Interval(2.8, 3.1)]), {pd.Interval(1.0, 2.7): pd.Interval(10.6, 20.8)}, IntervalArray([pd.Interval(10.6, 20.8), pd.Interval(2.8, 3.1)])), (pd.PeriodDtype('M'), [pd.Period('2020-05', freq='M')], {pd.Period('2020-05', freq='M'): pd.Period('2020-06', freq='M')}, [pd.Period('2020-06', freq='M')])])
def test_replace_dtype(self, dtype, input_data, to_replace, expected_data):
    ser = pd.Series(input_data, dtype=dtype)
    result = ser.replace(to_replace)
    expected = pd.Series(expected_data, dtype=dtype)
    tm.assert_series_equal(result, expected)