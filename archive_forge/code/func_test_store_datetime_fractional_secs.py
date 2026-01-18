import datetime
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
@pytest.mark.parametrize('unit', ['us', 'ns'])
def test_store_datetime_fractional_secs(setup_path, unit):
    dt = datetime.datetime(2012, 1, 2, 3, 4, 5, 123456)
    dti = DatetimeIndex([dt], dtype=f'M8[{unit}]')
    series = Series([0], index=dti)
    with ensure_clean_store(setup_path) as store:
        store['a'] = series
        assert store['a'].index[0] == dt