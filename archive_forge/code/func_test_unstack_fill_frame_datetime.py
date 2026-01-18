from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_fill_frame_datetime(self):
    dv = date_range('2012-01-01', periods=4).values
    data = Series(dv)
    data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = data.unstack()
    expected = DataFrame({'a': [dv[0], pd.NaT, dv[3]], 'b': [dv[1], dv[2], pd.NaT]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value=dv[0])
    expected = DataFrame({'a': [dv[0], dv[0], dv[3]], 'b': [dv[1], dv[2], dv[0]]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)