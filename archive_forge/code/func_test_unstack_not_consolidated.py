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
def test_unstack_not_consolidated(self, using_array_manager):
    df = DataFrame({'x': [1, 2, np.nan], 'y': [3.0, 4, np.nan]})
    df2 = df[['x']]
    df2['y'] = df['y']
    if not using_array_manager:
        assert len(df2._mgr.blocks) == 2
    res = df2.unstack()
    expected = df.unstack()
    tm.assert_series_equal(res, expected)