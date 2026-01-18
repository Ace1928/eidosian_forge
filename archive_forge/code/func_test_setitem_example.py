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
def test_setitem_example(self):
    idx = IntervalIndex.from_breaks(range(4))
    obj = Series(idx)
    val = Interval(0.5, 1.5)
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        obj[0] = val
    assert obj.dtype == 'Interval[float64, right]'