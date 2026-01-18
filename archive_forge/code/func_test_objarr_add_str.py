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
def test_objarr_add_str(self, box_with_array):
    ser = Series(['x', np.nan, 'x'])
    expected = Series(['xa', np.nan, 'xa'])
    ser = tm.box_expected(ser, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = ser + 'a'
    tm.assert_equal(result, expected)