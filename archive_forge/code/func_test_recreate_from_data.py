import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('freq', ['M', 'Q', 'Y', 'D', 'B', 'min', 's', 'ms', 'us', 'ns', 'h'])
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_recreate_from_data(self, freq):
    org = period_range(start='2001/04/01', freq=freq, periods=1)
    idx = PeriodIndex(org.values, freq=freq)
    tm.assert_index_equal(idx, org)