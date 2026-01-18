import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('data', [9.0, -9.0, 0.0])
@pytest.mark.parametrize('length', [1, 2])
def test_single_bin(data, length):
    ser = Series([data] * length)
    result = cut(ser, 1, labels=False)
    expected = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)