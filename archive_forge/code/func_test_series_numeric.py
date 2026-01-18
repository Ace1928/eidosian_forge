import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data', [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data):
    ser = Series(data, index=list('ABCD'), name='EFG')
    result = to_numeric(ser)
    tm.assert_series_equal(result, ser)