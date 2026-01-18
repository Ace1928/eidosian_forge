from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_single_column_mixed_datetime(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['foo', 'bar', 'baz'])
    df['timestamp'] = Timestamp('20010102')
    result = df.dtypes
    expected = Series([np.dtype('float64')] * 3 + [np.dtype('datetime64[s]')], index=['foo', 'bar', 'baz', 'timestamp'])
    tm.assert_series_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        df.loc['b', 'timestamp'] = iNaT
    assert not isna(df.loc['b', 'timestamp'])
    assert df['timestamp'].dtype == np.object_
    assert df.loc['b', 'timestamp'] == iNaT
    df.loc['c', 'timestamp'] = np.nan
    assert isna(df.loc['c', 'timestamp'])
    df.loc['d', :] = np.nan
    assert not isna(df.loc['c', :]).all()