import numpy as np
import pytest
from pandas.errors import DataError
from pandas.core.dtypes.common import pandas_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, data, expected_data, coerce_int, min_periods', [('count', np.arange(5), [1, 2, 2, 2, 2], True, 0), ('count', np.arange(10, 0, -2), [1, 2, 2, 2, 2], True, 0), ('count', [0, 1, 2, np.nan, 4], [1, 2, 2, 1, 1], False, 0), ('max', np.arange(5), [np.nan, 1, 2, 3, 4], True, None), ('max', np.arange(10, 0, -2), [np.nan, 10, 8, 6, 4], True, None), ('max', [0, 1, 2, np.nan, 4], [np.nan, 1, 2, np.nan, np.nan], False, None), ('min', np.arange(5), [np.nan, 0, 1, 2, 3], True, None), ('min', np.arange(10, 0, -2), [np.nan, 8, 6, 4, 2], True, None), ('min', [0, 1, 2, np.nan, 4], [np.nan, 0, 1, np.nan, np.nan], False, None), ('sum', np.arange(5), [np.nan, 1, 3, 5, 7], True, None), ('sum', np.arange(10, 0, -2), [np.nan, 18, 14, 10, 6], True, None), ('sum', [0, 1, 2, np.nan, 4], [np.nan, 1, 3, np.nan, np.nan], False, None), ('mean', np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None), ('mean', np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None), ('mean', [0, 1, 2, np.nan, 4], [np.nan, 0.5, 1.5, np.nan, np.nan], False, None), ('std', np.arange(5), [np.nan] + [np.sqrt(0.5)] * 4, True, None), ('std', np.arange(10, 0, -2), [np.nan] + [np.sqrt(2)] * 4, True, None), ('std', [0, 1, 2, np.nan, 4], [np.nan] + [np.sqrt(0.5)] * 2 + [np.nan] * 2, False, None), ('var', np.arange(5), [np.nan, 0.5, 0.5, 0.5, 0.5], True, None), ('var', np.arange(10, 0, -2), [np.nan, 2, 2, 2, 2], True, None), ('var', [0, 1, 2, np.nan, 4], [np.nan, 0.5, 0.5, np.nan, np.nan], False, None), ('median', np.arange(5), [np.nan, 0.5, 1.5, 2.5, 3.5], True, None), ('median', np.arange(10, 0, -2), [np.nan, 9, 7, 5, 3], True, None), ('median', [0, 1, 2, np.nan, 4], [np.nan, 0.5, 1.5, np.nan, np.nan], False, None)])
def test_series_dtypes(method, data, expected_data, coerce_int, dtypes, min_periods, step):
    ser = Series(data, dtype=get_dtype(dtypes, coerce_int=coerce_int))
    rolled = ser.rolling(2, min_periods=min_periods, step=step)
    if dtypes in ('m8[ns]', 'M8[ns]', 'datetime64[ns, UTC]') and method != 'count':
        msg = 'No numeric types to aggregate'
        with pytest.raises(DataError, match=msg):
            getattr(rolled, method)()
    else:
        result = getattr(rolled, method)()
        expected = Series(expected_data, dtype='float64')[::step]
        tm.assert_almost_equal(result, expected)