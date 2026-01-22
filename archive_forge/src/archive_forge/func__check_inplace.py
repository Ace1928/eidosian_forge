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
def _check_inplace(self, is_inplace, orig, arr, obj):
    if is_inplace is None:
        pass
    elif is_inplace:
        if arr.dtype.kind in ['m', 'M']:
            assert arr._ndarray is obj._values._ndarray
        else:
            assert obj._values is arr
    else:
        tm.assert_equal(arr, orig._values)