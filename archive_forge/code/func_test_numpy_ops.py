from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('numpy_op, expected', [(np.sum, 10), (np.nansum, 10), (np.mean, 2.5), (np.nanmean, 2.5), (np.median, 2.5), (np.nanmedian, 2.5), (np.min, 1), (np.max, 4), (np.nanmin, 1), (np.nanmax, 4)])
def test_numpy_ops(numpy_op, expected):
    result = numpy_op(Series([1, 2, 3, 4]))
    assert result == expected