import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_with_duplicates_no_on(self):
    df1 = pd.DataFrame({'key': [1, 1, 3], 'left_val': [1, 2, 3]})
    df2 = pd.DataFrame({'key': [1, 2, 2], 'right_val': [1, 2, 3]})
    result = merge_asof(df1, df2, on='key')
    expected = pd.DataFrame({'key': [1, 1, 3], 'left_val': [1, 2, 3], 'right_val': [1, 1, 3]})
    tm.assert_frame_equal(result, expected)