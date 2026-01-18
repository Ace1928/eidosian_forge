from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_setitem_with_expansion(self):
    df = DataFrame(data=to_datetime(['2015-03-30 20:12:32', '2015-03-12 00:11:11']), columns=['time'])
    df['new_col'] = ['new', 'old']
    df.time = df.set_index('time').index.tz_localize('UTC')
    v = df[df.new_col == 'new'].set_index('time').index.tz_convert('US/Pacific')
    df2 = df.copy()
    df2.loc[df2.new_col == 'new', 'time'] = v
    expected = Series([v[0].tz_convert('UTC'), df.loc[1, 'time']], name='time')
    tm.assert_series_equal(df2.time, expected)
    v = df.loc[df.new_col == 'new', 'time'] + Timedelta('1s')
    df.loc[df.new_col == 'new', 'time'] = v
    tm.assert_series_equal(df.loc[df.new_col == 'new', 'time'], v)