from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
@pytest.mark.parametrize('td', [Timedelta(hours=3), np.timedelta64(3, 'h'), timedelta(hours=3)])
def test_radd_tdscalar(self, td, fixed_now_ts):
    ts = fixed_now_ts
    assert td + ts == ts + td