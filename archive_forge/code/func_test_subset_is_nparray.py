import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subset_is_nparray(self):
    df = DataFrame({'A': [1, 2, np.nan], 'B': list('abc'), 'C': [4, np.nan, 5]})
    expected = DataFrame({'A': [1.0], 'B': ['a'], 'C': [4.0]})
    result = df.dropna(subset=np.array(['A', 'C']))
    tm.assert_frame_equal(result, expected)