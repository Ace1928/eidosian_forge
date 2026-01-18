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
@pytest.mark.parametrize('initial', ['2018-10-08 13:36:45+00:00', '2018-10-08 13:36:45+03:00'])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_preserve_timezone(self, initial: str, method):
    initial_dt = to_datetime(initial)
    expected = Series([initial_dt])
    df = DataFrame([expected])
    result = getattr(df, method)(axis=1)
    tm.assert_series_equal(result, expected)