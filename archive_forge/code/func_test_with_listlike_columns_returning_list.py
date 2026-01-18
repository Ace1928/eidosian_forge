from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_with_listlike_columns_returning_list():
    df = DataFrame({'x': Series([['a', 'b'], ['q']]), 'y': Series([['z'], ['q', 't']])})
    df.index = MultiIndex.from_tuples([('i0', 'j0'), ('i1', 'j1')])
    result = df.apply(lambda row: [el for el in row['x'] if el in row['y']], axis=1)
    expected = Series([[], ['q']], index=df.index)
    tm.assert_series_equal(result, expected)