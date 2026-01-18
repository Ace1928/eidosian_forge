from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('fold', [0, 1])
def test_replace_preserves_fold(fold):
    tz = gettz('Europe/Moscow')
    ts = Timestamp(year=2009, month=10, day=25, hour=2, minute=30, fold=fold, tzinfo=tz)
    ts_replaced = ts.replace(second=1)
    assert ts_replaced.fold == fold