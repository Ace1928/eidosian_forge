import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('q', [2, 5, 10])
def test_qcut_nullable_integer(q, any_numeric_ea_dtype):
    arr = pd.array(np.arange(100), dtype=any_numeric_ea_dtype)
    arr[::2] = pd.NA
    result = qcut(arr, q)
    expected = qcut(arr.astype(float), q)
    tm.assert_categorical_equal(result, expected)