import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('freq', ['2BME', '2YE-MAR', '2QE'])
def test_pi_asfreq_invalid_frequency(self, freq):
    msg = f'Invalid frequency: {freq}'
    pi = PeriodIndex(['2020-01-01', '2021-01-01'], freq='M')
    with pytest.raises(ValueError, match=msg):
        pi.asfreq(freq=freq)