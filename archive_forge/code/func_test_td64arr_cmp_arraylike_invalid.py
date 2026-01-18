from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.date_range('1970-01-01', periods=10, tz='UTC').array, np.array(pd.date_range('1970-01-01', periods=10)), list(pd.date_range('1970-01-01', periods=10)), pd.date_range('1970-01-01', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
def test_td64arr_cmp_arraylike_invalid(self, other, box_with_array):
    rng = timedelta_range('1 days', periods=10)._data
    rng = tm.box_expected(rng, box_with_array)
    assert_invalid_comparison(rng, other, box_with_array)