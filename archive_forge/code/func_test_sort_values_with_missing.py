from copy import (
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
@pytest.mark.parametrize('na_position', ['first', 'last'])
def test_sort_values_with_missing(index_with_missing, na_position, request):
    if isinstance(index_with_missing, CategoricalIndex):
        request.applymarker(pytest.mark.xfail(reason='missing value sorting order not well-defined', strict=False))
    missing_count = np.sum(index_with_missing.isna())
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    sorted_values = np.sort(not_na_vals)
    if na_position == 'first':
        sorted_values = np.concatenate([[None] * missing_count, sorted_values])
    else:
        sorted_values = np.concatenate([sorted_values, [None] * missing_count])
    expected = type(index_with_missing)(sorted_values, dtype=index_with_missing.dtype)
    result = index_with_missing.sort_values(na_position=na_position)
    tm.assert_index_equal(result, expected)