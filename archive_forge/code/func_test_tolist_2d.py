import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
def test_tolist_2d(self, data):
    arr2d = data.reshape(1, -1)
    result = arr2d.tolist()
    expected = [data.tolist()]
    assert isinstance(result, list)
    assert all((isinstance(x, list) for x in result))
    assert result == expected