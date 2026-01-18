from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('to_key', ['timedelta64[ns]', 'bool', 'object', 'complex128', 'float64', 'int64'], indirect=True)
@pytest.mark.parametrize('from_key', ['datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
def test_replace_series_datetime_tz(self, how, to_key, from_key, replacer, using_infer_string):
    index = pd.Index([3, 4], name='xyz')
    obj = pd.Series(self.rep[from_key], index=index, name='yyy')
    assert obj.dtype == from_key
    exp = pd.Series(self.rep[to_key], index=index, name='yyy')
    if using_infer_string and to_key == 'object':
        assert exp.dtype == 'string'
    else:
        assert exp.dtype == to_key
    msg = 'Downcasting behavior in `replace`'
    warn = FutureWarning if exp.dtype != object else None
    with tm.assert_produces_warning(warn, match=msg):
        result = obj.replace(replacer)
    tm.assert_series_equal(result, exp)