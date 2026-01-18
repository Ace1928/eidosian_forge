import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_values_multiplying_large_series_by_NA():
    result = pd.NA * pd.Series(np.zeros(10001))
    expected = pd.Series([pd.NA] * 10001)
    tm.assert_series_equal(result, expected)