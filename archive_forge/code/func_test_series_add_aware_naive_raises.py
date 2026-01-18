from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_series_add_aware_naive_raises(self):
    rng = date_range('1/1/2011', periods=10, freq='h')
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ser_utc = ser.tz_localize('utc')
    msg = 'Cannot join tz-naive with tz-aware DatetimeIndex'
    with pytest.raises(Exception, match=msg):
        ser + ser_utc
    with pytest.raises(Exception, match=msg):
        ser_utc + ser