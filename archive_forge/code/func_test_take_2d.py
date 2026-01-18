import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def test_take_2d(self, data):
    arr2d = data.reshape(-1, 1)
    result = arr2d.take([0, 0, -1], axis=0)
    expected = data.take([0, 0, -1]).reshape(-1, 1)
    tm.assert_extension_array_equal(result, expected)