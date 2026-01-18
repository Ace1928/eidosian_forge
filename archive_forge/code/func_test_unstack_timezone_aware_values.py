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
def test_unstack_timezone_aware_values():
    df = DataFrame({'timestamp': [pd.Timestamp('2017-08-27 01:00:00.709949+0000', tz='UTC')], 'a': ['a'], 'b': ['b'], 'c': ['c']}, columns=['timestamp', 'a', 'b', 'c'])
    result = df.set_index(['a', 'b']).unstack()
    expected = DataFrame([[pd.Timestamp('2017-08-27 01:00:00.709949+0000', tz='UTC'), 'c']], index=Index(['a'], name='a'), columns=MultiIndex(levels=[['timestamp', 'c'], ['b']], codes=[[0, 1], [0, 0]], names=[None, 'b']))
    tm.assert_frame_equal(result, expected)