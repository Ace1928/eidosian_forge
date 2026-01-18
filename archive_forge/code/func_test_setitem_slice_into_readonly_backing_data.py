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
def test_setitem_slice_into_readonly_backing_data():
    array = np.zeros(5)
    array.flags.writeable = False
    series = Series(array, copy=False)
    msg = 'assignment destination is read-only'
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1
    assert not array.any()