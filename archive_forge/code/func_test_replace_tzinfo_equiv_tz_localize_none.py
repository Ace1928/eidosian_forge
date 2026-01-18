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
def test_replace_tzinfo_equiv_tz_localize_none(self):
    ts = Timestamp('2013-11-03 01:59:59.999999-0400', tz='US/Eastern')
    assert ts.tz_localize(None) == ts.replace(tzinfo=None)