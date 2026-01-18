from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_operators_datetimelike(self):
    td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
    td1.iloc[2] = np.nan
    dt1 = Series([Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')])
    dt1.iloc[2] = np.nan
    dt2 = Series([Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')])
    dt1 - dt2
    dt2 - dt1
    dt1 + td1
    td1 + dt1
    dt1 - td1
    td1 + dt1
    dt1 + td1