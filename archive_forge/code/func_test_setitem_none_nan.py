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
def test_setitem_none_nan(self):
    series = Series(date_range('1/1/2000', periods=10))
    series[3] = None
    assert series[3] is NaT
    series[3:5] = None
    assert series[4] is NaT
    series[5] = np.nan
    assert series[5] is NaT
    series[5:7] = np.nan
    assert series[6] is NaT