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
def test_unstack_categorical_columns(self):
    idx = MultiIndex.from_product([['A'], [0, 1]])
    df = DataFrame({'cat': pd.Categorical(['a', 'b'])}, index=idx)
    result = df.unstack()
    expected = DataFrame({0: pd.Categorical(['a'], categories=['a', 'b']), 1: pd.Categorical(['b'], categories=['a', 'b'])}, index=['A'])
    expected.columns = MultiIndex.from_tuples([('cat', 0), ('cat', 1)])
    tm.assert_frame_equal(result, expected)