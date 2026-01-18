from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('array, dtype', [(['a', 'b'], object), (pd.period_range('12-1-2000', periods=2, freq='Q-DEC'), pd.PeriodDtype(freq='Q-DEC'))])
def test_reset_index_dtypes_on_empty_series_with_multiindex(array, dtype, using_infer_string):
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    result = Series(dtype=object, index=idx)[:0].reset_index().dtypes
    exp = 'string' if using_infer_string else object
    expected = Series({'level_0': np.int64, 'level_1': np.float64, 'level_2': exp if dtype == object else dtype, 0: object})
    tm.assert_series_equal(result, expected)