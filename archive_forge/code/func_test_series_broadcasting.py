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
def test_series_broadcasting(self):
    df = DataFrame([1.0, 1.0, 1.0])
    df_nan = DataFrame({'A': [np.nan, 2.0, np.nan]})
    s = Series([1, 1, 1])
    s_nan = Series([np.nan, np.nan, 1])
    with tm.assert_produces_warning(None):
        df_nan.clip(lower=s, axis=0)
        for op in ['lt', 'le', 'gt', 'ge', 'eq', 'ne']:
            getattr(df, op)(s_nan, axis=0)