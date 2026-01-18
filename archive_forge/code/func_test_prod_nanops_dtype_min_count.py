from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('float_type', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('kwargs, expected_result', [({'axis': 1, 'min_count': 2}, [2.0, 4.0, np.nan]), ({'axis': 1, 'min_count': 3}, [np.nan, np.nan, np.nan]), ({'axis': 1, 'skipna': False}, [2.0, 4.0, np.nan])])
def test_prod_nanops_dtype_min_count(self, float_type, kwargs, expected_result):
    df = DataFrame({'a': [1.0, 2.0, 4.4], 'b': [2.0, 2.0, np.nan]}, dtype=float_type)
    result = df.prod(**kwargs)
    expected = Series(expected_result).astype(float_type)
    tm.assert_series_equal(result, expected)