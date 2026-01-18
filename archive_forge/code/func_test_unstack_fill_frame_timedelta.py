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
def test_unstack_fill_frame_timedelta(self):
    td = [Timedelta(days=i) for i in range(4)]
    data = Series(td)
    data.index = MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'b'), ('z', 'a')])
    result = data.unstack()
    expected = DataFrame({'a': [td[0], pd.NaT, td[3]], 'b': [td[1], td[2], pd.NaT]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)
    result = data.unstack(fill_value=td[1])
    expected = DataFrame({'a': [td[0], td[1], td[3]], 'b': [td[1], td[2], td[1]]}, index=['x', 'y', 'z'])
    tm.assert_frame_equal(result, expected)