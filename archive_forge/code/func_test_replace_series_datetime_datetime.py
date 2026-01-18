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
@pytest.mark.parametrize('to_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
@pytest.mark.parametrize('from_key', ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, US/Eastern]'], indirect=True)
def test_replace_series_datetime_datetime(self, how, to_key, from_key, replacer):
    index = pd.Index([3, 4], name='xyz')
    obj = pd.Series(self.rep[from_key], index=index, name='yyy')
    assert obj.dtype == from_key
    exp = pd.Series(self.rep[to_key], index=index, name='yyy')
    warn = FutureWarning
    if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(exp.dtype, pd.DatetimeTZDtype):
        exp = exp.astype(obj.dtype)
        warn = None
    else:
        assert exp.dtype == to_key
        if to_key == from_key:
            warn = None
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(warn, match=msg):
        result = obj.replace(replacer)
    tm.assert_series_equal(result, exp)