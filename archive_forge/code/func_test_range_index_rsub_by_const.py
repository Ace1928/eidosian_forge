import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_range_index_rsub_by_const(self):
    result = 3 - RangeIndex(0, 4, 1)
    expected = RangeIndex(3, -1, -1)
    tm.assert_index_equal(result, expected)