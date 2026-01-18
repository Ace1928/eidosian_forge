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
def test_setitem_boolean_different_order(self, string_series):
    ordered = string_series.sort_values()
    copy = string_series.copy()
    copy[ordered > 0] = 0
    expected = string_series.copy()
    expected[expected > 0] = 0
    tm.assert_series_equal(copy, expected)