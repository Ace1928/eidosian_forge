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
def test_setitem_nan_into_categorical():
    ser = Series(Categorical([1, 2, 3]))
    exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)