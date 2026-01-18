import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace2(self):
    N = 50
    ser = pd.Series(np.fabs(np.random.default_rng(2).standard_normal(N)), pd.date_range('2020-01-01', periods=N), dtype=object)
    ser[:5] = np.nan
    ser[6:10] = 'foo'
    ser[20:30] = 'bar'
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.replace([np.nan, 'foo', 'bar'], -1)
    assert (rs[:5] == -1).all()
    assert (rs[6:10] == -1).all()
    assert (rs[20:30] == -1).all()
    assert pd.isna(ser[:5]).all()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.replace({np.nan: -1, 'foo': -2, 'bar': -3})
    assert (rs[:5] == -1).all()
    assert (rs[6:10] == -2).all()
    assert (rs[20:30] == -3).all()
    assert pd.isna(ser[:5]).all()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs2 = ser.replace([np.nan, 'foo', 'bar'], [-1, -2, -3])
    tm.assert_series_equal(rs, rs2)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        return_value = ser.replace([np.nan, 'foo', 'bar'], -1, inplace=True)
    assert return_value is None
    assert (ser[:5] == -1).all()
    assert (ser[6:10] == -1).all()
    assert (ser[20:30] == -1).all()