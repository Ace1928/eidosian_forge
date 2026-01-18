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
@pytest.mark.parametrize('kwd', sorted(liboffsets._relativedelta_kwds))
def test_valid_relativedelta_kwargs(kwd, request):
    if kwd == 'millisecond':
        request.applymarker(pytest.mark.xfail(raises=NotImplementedError, reason='Constructing DateOffset object with `millisecond` is not yet supported.'))
    DateOffset(**{kwd: 1})