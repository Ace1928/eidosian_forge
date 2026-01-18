import datetime
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_tseries_indices_series(setup_path):
    with ensure_clean_store(setup_path) as store:
        idx = date_range('2020-01-01', periods=10)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        store['a'] = ser
        result = store['a']
        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj='series index')
        idx = period_range('2020-01-01', periods=10, freq='D')
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx)
        store['a'] = ser
        result = store['a']
        tm.assert_series_equal(result, ser)
        assert result.index.freq == ser.index.freq
        tm.assert_class_equal(result.index, ser.index, obj='series index')