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
@pytest.mark.skipif(using_pyarrow_string_dtype(), reason='TODO: test is to complex')
def test_replace_series(self, how, to_key, from_key, replacer):
    index = pd.Index([3, 4], name='xxx')
    obj = pd.Series(self.rep[from_key], index=index, name='yyy')
    assert obj.dtype == from_key
    if from_key.startswith('datetime') and to_key.startswith('datetime'):
        return
    elif from_key in ['datetime64[ns, US/Eastern]', 'datetime64[ns, UTC]']:
        return
    if from_key == 'float64' and to_key in 'int64' or (from_key == 'complex128' and to_key in ('int64', 'float64')):
        if not IS64 or is_platform_windows():
            pytest.skip(f'32-bit platform buggy: {from_key} -> {to_key}')
        exp = pd.Series(self.rep[to_key], index=index, name='yyy', dtype=from_key)
    else:
        exp = pd.Series(self.rep[to_key], index=index, name='yyy')
        assert exp.dtype == to_key
    msg = 'Downcasting behavior in `replace`'
    warn = FutureWarning
    if exp.dtype == obj.dtype or exp.dtype == object or (exp.dtype.kind in 'iufc' and obj.dtype.kind in 'iufc'):
        warn = None
    with tm.assert_produces_warning(warn, match=msg):
        result = obj.replace(replacer)
    tm.assert_series_equal(result, exp)