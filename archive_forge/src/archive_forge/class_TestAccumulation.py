import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
class TestAccumulation(base.BaseAccumulateTests):

    def check_accumulate(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        expected = getattr(pd.Series(s.astype('float64')), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)
        if op_name in ('cummin', 'cummax'):
            assert is_bool_dtype(result)

    @pytest.mark.parametrize('skipna', [True, False])
    def test_accumulate_series_raises(self, data, all_numeric_accumulations, skipna):
        pass