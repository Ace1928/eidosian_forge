import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_string_with_number(self):
    s = pd.Series([1, 2, 3])
    result = s.replace('2', np.nan)
    expected = pd.Series([1, 2, 3])
    tm.assert_series_equal(expected, result)