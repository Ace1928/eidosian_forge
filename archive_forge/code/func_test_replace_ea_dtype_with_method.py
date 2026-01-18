import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_ea_dtype_with_method(self, any_numeric_ea_dtype):
    arr = pd.array([1, 2, pd.NA, 4], dtype=any_numeric_ea_dtype)
    ser = pd.Series(arr)
    self._check_replace_with_method(ser)