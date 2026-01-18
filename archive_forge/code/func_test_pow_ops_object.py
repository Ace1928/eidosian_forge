import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_pow_ops_object(self):
    a = Series([1, np.nan, 1, np.nan], dtype=object)
    b = Series([1, np.nan, np.nan, 1], dtype=object)
    result = a ** b
    expected = Series(a.values ** b.values, dtype=object)
    tm.assert_series_equal(result, expected)
    result = b ** a
    expected = Series(b.values ** a.values, dtype=object)
    tm.assert_series_equal(result, expected)