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
@pytest.mark.parametrize('values', [['2022-01-01', '2022-01-02', pd.NaT, '2022-01-03'], 4 * [pd.NaT]])
def test_std_datetime64_with_nat(self, values, skipna, using_array_manager, request, unit):
    if using_array_manager and (not skipna or all((value is pd.NaT for value in values))):
        mark = pytest.mark.xfail(reason='GH#51446: Incorrect type inference on NaT in reduction result')
        request.applymarker(mark)
    dti = to_datetime(values).as_unit(unit)
    df = DataFrame({'a': dti})
    result = df.std(skipna=skipna)
    if not skipna or all((value is pd.NaT for value in values)):
        expected = Series({'a': pd.NaT}, dtype=f'timedelta64[{unit}]')
    else:
        expected = Series({'a': 86400000000000}, dtype=f'timedelta64[{unit}]')
    tm.assert_series_equal(result, expected)