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
def test_setitem_invalidates_datetime_index_freq(self, using_copy_on_write):
    dti = date_range('20130101', periods=3, tz='US/Eastern')
    ts = dti[1]
    ser = Series(dti)
    assert ser._values is not dti
    if using_copy_on_write:
        assert ser._values._ndarray.base is dti._data._ndarray.base
    else:
        assert ser._values._ndarray.base is not dti._data._ndarray.base
    assert dti.freq == 'D'
    ser.iloc[1] = NaT
    assert ser._values.freq is None
    assert ser._values is not dti
    assert ser._values._ndarray.base is not dti._data._ndarray.base
    assert dti[1] == ts
    assert dti.freq == 'D'