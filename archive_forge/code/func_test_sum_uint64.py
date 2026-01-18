from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_sum_uint64(self):
    s = Series([10000000000000000000], dtype='uint64')
    result = s.sum()
    expected = np.uint64(10000000000000000000)
    tm.assert_almost_equal(result, expected)