import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_empty_series_and_empty_cond_having_non_bool_dtypes():
    ser = Series([], dtype=float)
    result = ser.where([])
    tm.assert_series_equal(result, ser)