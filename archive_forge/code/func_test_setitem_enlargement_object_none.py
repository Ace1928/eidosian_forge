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
def test_setitem_enlargement_object_none(self, nulls_fixture, using_infer_string):
    ser = Series(['a', 'b'])
    ser[3] = nulls_fixture
    dtype = 'string[pyarrow_numpy]' if using_infer_string and (not isinstance(nulls_fixture, Decimal)) else object
    expected = Series(['a', 'b', nulls_fixture], index=[0, 1, 3], dtype=dtype)
    tm.assert_series_equal(ser, expected)
    if using_infer_string:
        ser[3] is np.nan
    else:
        assert ser[3] is nulls_fixture