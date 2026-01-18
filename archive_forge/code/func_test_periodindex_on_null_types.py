import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('null_val', [None, np.nan, NaT, NA, math.nan, 'NaT', 'nat', 'NAT', 'nan', 'NaN', 'NAN'])
def test_periodindex_on_null_types(self, null_val):
    result = PeriodIndex(['2022-04-06', '2022-04-07', null_val], freq='D')
    expected = PeriodIndex(['2022-04-06', '2022-04-07', 'NaT'], dtype='period[D]')
    assert result[2] is NaT
    tm.assert_index_equal(result, expected)