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
@pytest.mark.parametrize('size', range(2, 6))
@pytest.mark.parametrize('mask', [[True, False, False, False, False], [True, False], [False]])
@pytest.mark.parametrize('item', [2.0, np.nan, np.finfo(float).max, np.finfo(float).min])
@pytest.mark.parametrize('box', [lambda x: np.array([x]), lambda x: [x], lambda x: (x,)])
def test_setitem_bool_indexer_dont_broadcast_length1_values(size, mask, item, box):
    selection = np.resize(mask, size)
    data = np.arange(size, dtype=float)
    ser = Series(data)
    if selection.sum() != 1:
        msg = 'cannot set using a list-like indexer with a different length than the value'
        with pytest.raises(ValueError, match=msg):
            ser[selection] = box(item)
    else:
        ser[selection] = box(item)
        expected = Series(np.arange(size, dtype=float))
        expected[selection] = item
        tm.assert_series_equal(ser, expected)