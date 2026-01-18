import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('values, expected', [(['a', 'b', 'c'], np.array([False, False, False])), (['a', 'b', None], np.array([False, False, True]))])
def test_use_inf_as_na(values, expected, dtype):
    values = pd.array(values, dtype=dtype)
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            result = values.isna()
            tm.assert_numpy_array_equal(result, expected)
            result = pd.Series(values).isna()
            expected = pd.Series(expected)
            tm.assert_series_equal(result, expected)
            result = pd.DataFrame(values).isna()
            expected = pd.DataFrame(expected)
            tm.assert_frame_equal(result, expected)