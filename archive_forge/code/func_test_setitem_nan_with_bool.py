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
def test_setitem_nan_with_bool(self):
    result = Series([True, False, True])
    with tm.assert_produces_warning(FutureWarning, match='item of incompatible dtype'):
        result[0] = np.nan
    expected = Series([np.nan, False, True], dtype=object)
    tm.assert_series_equal(result, expected)