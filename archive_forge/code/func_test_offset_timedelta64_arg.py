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
def test_offset_timedelta64_arg(self, offset_types):
    off = _create_offset(offset_types)
    td64 = np.timedelta64(4567, 's')
    with pytest.raises(TypeError, match='argument must be an integer'):
        type(off)(n=td64, **off.kwds)