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
def test_setitem_boolean_corner(self, datetime_series):
    ts = datetime_series
    mask_shifted = ts.shift(1, freq=BDay()) > ts.median()
    msg = 'Unalignable boolean Series provided as indexer \\(index of the boolean Series and of the indexed object do not match'
    with pytest.raises(IndexingError, match=msg):
        ts[mask_shifted] = 1
    with pytest.raises(IndexingError, match=msg):
        ts.loc[mask_shifted] = 1