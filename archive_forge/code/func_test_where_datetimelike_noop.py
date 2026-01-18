from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('dtype', ['timedelta64[ns]', 'datetime64[ns]', 'datetime64[ns, Asia/Tokyo]', 'Period[D]'])
def test_where_datetimelike_noop(self, dtype):
    with tm.assert_produces_warning(FutureWarning, match='is deprecated'):
        ser = Series(np.arange(3) * 10 ** 9, dtype=np.int64).view(dtype)
    df = ser.to_frame()
    mask = np.array([False, False, False])
    res = ser.where(~mask, 'foo')
    tm.assert_series_equal(res, ser)
    mask2 = mask.reshape(-1, 1)
    res2 = df.where(~mask2, 'foo')
    tm.assert_frame_equal(res2, df)
    res3 = ser.mask(mask, 'foo')
    tm.assert_series_equal(res3, ser)
    res4 = df.mask(mask2, 'foo')
    tm.assert_frame_equal(res4, df)
    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res5 = df.where(mask2, 4)
    expected = DataFrame(4, index=df.index, columns=df.columns)
    tm.assert_frame_equal(res5, expected)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.mask(~mask2, 4, inplace=True)
    tm.assert_frame_equal(df, expected.astype(object))