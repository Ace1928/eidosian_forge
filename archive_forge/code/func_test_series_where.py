from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_series_where(self, obj, key, expected, warn, val, is_inplace):
    mask = np.zeros(obj.shape, dtype=bool)
    mask[key] = True
    if is_list_like(val) and len(val) < len(obj):
        msg = 'operands could not be broadcast together with shapes'
        with pytest.raises(ValueError, match=msg):
            obj.where(~mask, val)
        return
    orig = obj
    obj = obj.copy()
    arr = obj._values
    res = obj.where(~mask, val)
    if val is NA and res.dtype == object:
        expected = expected.fillna(NA)
    elif val is None and res.dtype == object:
        assert expected.dtype == object
        expected = expected.copy()
        expected[expected.isna()] = None
    tm.assert_series_equal(res, expected)
    self._check_inplace(is_inplace, orig, arr, obj)