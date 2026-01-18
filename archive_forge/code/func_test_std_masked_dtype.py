import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_std_masked_dtype(any_numeric_ea_dtype):
    df = DataFrame({'a': [2, 1, 1, 1, 2, 2, 1], 'b': Series([pd.NA, 1, 2, 1, 1, 1, 2], dtype='Float64')})
    result = df.groupby('a').std()
    expected = DataFrame({'b': [0.57735, 0]}, index=Index([1, 2], name='a'), dtype='Float64')
    tm.assert_frame_equal(result, expected)