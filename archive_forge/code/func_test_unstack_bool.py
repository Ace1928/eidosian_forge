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
def test_unstack_bool(self):
    df = DataFrame([False, False], index=MultiIndex.from_arrays([['a', 'b'], ['c', 'l']]), columns=['col'])
    rs = df.unstack()
    xp = DataFrame(np.array([[False, np.nan], [np.nan, False]], dtype=object), index=['a', 'b'], columns=MultiIndex.from_arrays([['col', 'col'], ['c', 'l']]))
    tm.assert_frame_equal(rs, xp)