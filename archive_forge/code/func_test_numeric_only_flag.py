from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('meth', ['sem', 'var', 'std'])
def test_numeric_only_flag(self, meth):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
    df1 = df1.astype({'foo': object})
    df1.loc[0, 'foo'] = '100'
    df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
    df2 = df2.astype({'foo': object})
    df2.loc[0, 'foo'] = 'a'
    result = getattr(df1, meth)(axis=1, numeric_only=True)
    expected = getattr(df1[['bar', 'baz']], meth)(axis=1)
    tm.assert_series_equal(expected, result)
    result = getattr(df2, meth)(axis=1, numeric_only=True)
    expected = getattr(df2[['bar', 'baz']], meth)(axis=1)
    tm.assert_series_equal(expected, result)
    msg = "unsupported operand type\\(s\\) for -: 'float' and 'str'"
    with pytest.raises(TypeError, match=msg):
        getattr(df1, meth)(axis=1, numeric_only=False)
    msg = "could not convert string to float: 'a'"
    with pytest.raises(TypeError, match=msg):
        getattr(df2, meth)(axis=1, numeric_only=False)