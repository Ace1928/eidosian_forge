import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_size_series_dataframe():
    df = DataFrame(columns=['A', 'B'])
    out = Series(dtype='int64', index=Index([], name='A'))
    tm.assert_series_equal(df.groupby('A').size(), out)