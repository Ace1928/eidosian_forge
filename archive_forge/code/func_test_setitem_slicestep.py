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
def test_setitem_slicestep(self):
    series = Series(np.arange(20, dtype=np.float64), index=np.arange(20, dtype=np.int64))
    series[::2] = 0
    assert (series[::2] == 0).all()