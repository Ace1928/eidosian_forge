from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.int64, np.float32, np.float64, getattr(np, 'float128', None)])
@pytest.mark.parametrize('method', ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max'])
def test_returned_dtype(disable_bottleneck, dtype, method):
    if dtype is None:
        pytest.skip('np.float128 not available')
    ser = Series(range(10), dtype=dtype)
    result = getattr(ser, method)()
    if is_integer_dtype(dtype) and method not in ['min', 'max']:
        assert result.dtype == np.float64
    else:
        assert result.dtype == dtype