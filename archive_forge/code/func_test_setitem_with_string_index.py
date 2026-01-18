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
def test_setitem_with_string_index(self):
    ser = Series([1, 2, 3], index=['Date', 'b', 'other'], dtype=object)
    ser['Date'] = date.today()
    assert ser.Date == date.today()
    assert ser['Date'] == date.today()