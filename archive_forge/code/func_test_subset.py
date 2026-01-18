from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_subset():
    df = DataFrame({'c1': ['a', 'b', 'c'], 'c2': ['x', 'y', 'y']}, index=[0, 1, 1])
    result = df.groupby(level=0).value_counts(subset=['c2'])
    expected = Series([1, 2], index=MultiIndex.from_arrays([[0, 1], ['x', 'y']], names=[None, 'c2']), name='count')
    tm.assert_series_equal(result, expected)