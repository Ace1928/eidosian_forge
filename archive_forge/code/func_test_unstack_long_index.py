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
def test_unstack_long_index(self):
    df = DataFrame([[1]], columns=MultiIndex.from_tuples([[0]], names=['c1']), index=MultiIndex.from_tuples([[0, 0, 1, 0, 0, 0, 1]], names=['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']))
    result = df.unstack(['i2', 'i3', 'i4', 'i5', 'i6', 'i7'])
    expected = DataFrame([[1]], columns=MultiIndex.from_tuples([[0, 0, 1, 0, 0, 0, 1]], names=['c1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7']), index=Index([0], name='i1'))
    tm.assert_frame_equal(result, expected)