import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_to_array_integer():
    result = pd.array([1, 2], dtype='Float64')
    expected = pd.array([1.0, 2.0], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([1, 2], dtype='int32'), dtype='Float64')
    assert result.dtype == Float64Dtype()