import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('freq_depr', ['2SME', '2sme', '2CBME', '2BYE', '2Bye'])
def test_period_index_frequency_invalid_freq(self, freq_depr):
    msg = f'Invalid frequency: {freq_depr[1:]}'
    with pytest.raises(ValueError, match=msg):
        period_range('2020-01', '2020-05', freq=freq_depr)
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(['2020-01', '2020-05'], freq=freq_depr)