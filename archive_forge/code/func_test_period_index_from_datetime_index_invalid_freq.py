import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('freq', ['2BQE-SEP', '2BYE-MAR', '2BME'])
def test_period_index_from_datetime_index_invalid_freq(self, freq):
    msg = f'Invalid frequency: {freq[1:]}'
    rng = date_range('01-Jan-2012', periods=8, freq=freq)
    with pytest.raises(ValueError, match=msg):
        rng.to_period()