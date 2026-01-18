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
def test_setitem_negative_out_of_bounds(self):
    ser = Series(['a'] * 10, index=['a'] * 10)
    msg = 'index -11|-1 is out of bounds for axis 0 with size 10'
    warn_msg = 'Series.__setitem__ treating keys as positions is deprecated'
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            ser[-11] = 'foo'