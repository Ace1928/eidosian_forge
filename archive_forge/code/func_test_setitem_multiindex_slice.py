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
def test_setitem_multiindex_slice(self, indexer_sli):
    mi = MultiIndex.from_product(([0, 1], list('abcde')))
    result = Series(np.arange(10, dtype=np.int64), mi)
    indexer_sli(result)[::4] = 100
    expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
    tm.assert_series_equal(result, expected)