from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestAsOfLocs:

    def test_asof_locs_mismatched_type(self):
        dti = date_range('2016-01-01', periods=3)
        pi = dti.to_period('D')
        pi2 = dti.to_period('h')
        mask = np.array([0, 1, 0], dtype=bool)
        msg = 'must be DatetimeIndex or PeriodIndex'
        with pytest.raises(TypeError, match=msg):
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.int64), mask)
        with pytest.raises(TypeError, match=msg):
            pi.asof_locs(pd.Index(pi.asi8, dtype=np.float64), mask)
        with pytest.raises(TypeError, match=msg):
            pi.asof_locs(dti - dti, mask)
        msg = 'Input has different freq=h'
        with pytest.raises(libperiod.IncompatibleFrequency, match=msg):
            pi.asof_locs(pi2, mask)