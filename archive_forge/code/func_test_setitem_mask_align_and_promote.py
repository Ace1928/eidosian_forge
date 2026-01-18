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
def test_setitem_mask_align_and_promote(self):
    ts = Series(np.random.default_rng(2).standard_normal(100), index=np.arange(100, 0, -1)).round(5)
    mask = ts > 0
    left = ts.copy()
    right = ts[mask].copy().map(str)
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        left[mask] = right
    expected = ts.map(lambda t: str(t) if t > 0 else t)
    tm.assert_series_equal(left, expected)