from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_dt64tz(self, timezone_frame, using_copy_on_write):
    df = timezone_frame
    idx = df['B'].rename('foo')
    df['C'] = idx
    tm.assert_series_equal(df['C'], Series(idx, name='C'))
    df['D'] = 'foo'
    df['D'] = idx
    tm.assert_series_equal(df['D'], Series(idx, name='D'))
    del df['D']
    v1 = df._mgr.arrays[1]
    v2 = df._mgr.arrays[2]
    tm.assert_extension_array_equal(v1, v2)
    v1base = v1._ndarray.base
    v2base = v2._ndarray.base
    if not using_copy_on_write:
        assert v1base is None or id(v1base) != id(v2base)
    else:
        assert id(v1base) == id(v2base)
    df2 = df.copy()
    df2.iloc[1, 1] = NaT
    df2.iloc[1, 2] = NaT
    result = df2['B']
    tm.assert_series_equal(notna(result), Series([True, False, True], name='B'))
    tm.assert_series_equal(df2.dtypes, df.dtypes)