import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [9876543210.0, 2.0 ** 128])
def test_to_numeric_large_float_not_downcast_to_float_32(val):
    expected = Series([val])
    result = to_numeric(expected, downcast='float')
    tm.assert_series_equal(result, expected)