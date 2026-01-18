from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
def test_validate_n_error():
    with pytest.raises(TypeError, match='argument must be an integer'):
        DateOffset(n='Doh!')
    with pytest.raises(TypeError, match='argument must be an integer'):
        MonthBegin(n=timedelta(1))
    with pytest.raises(TypeError, match='argument must be an integer'):
        BDay(n=np.array([1, 2], dtype=np.int64))