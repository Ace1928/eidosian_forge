import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_use_inf_as_na_no_effect(self, data_missing):
    ser = pd.Series(data_missing)
    expected = ser.isna()
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            result = ser.isna()
    tm.assert_series_equal(result, expected)