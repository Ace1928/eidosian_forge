from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('ddof, exp', [(1, 2.5), (0, 2.0)])
def test_var_masked_array(self, ddof, exp):
    ser = Series([1, 2, 3, 4, 5], dtype='Int64')
    ser_numpy_dtype = Series([1, 2, 3, 4, 5], dtype='int64')
    result = ser.var(ddof=ddof)
    result_numpy_dtype = ser_numpy_dtype.var(ddof=ddof)
    assert result == result_numpy_dtype
    assert result == exp