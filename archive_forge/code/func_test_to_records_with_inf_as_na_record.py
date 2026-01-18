from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_records_with_inf_as_na_record(self):
    expected = '   NaN  inf         record\n0  inf    b    [0, inf, b]\n1  NaN  NaN  [1, nan, nan]\n2    e    f      [2, e, f]'
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with option_context('use_inf_as_na', True):
            df = DataFrame([[np.inf, 'b'], [np.nan, np.nan], ['e', 'f']], columns=[np.nan, np.inf])
            df['record'] = df[[np.nan, np.inf]].to_records()
            result = repr(df)
    assert result == expected