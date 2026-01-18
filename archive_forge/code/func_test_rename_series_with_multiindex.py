from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_series_with_multiindex(self):
    arrays = [['bar', 'baz', 'baz', 'foo', 'qux'], ['one', 'one', 'two', 'two', 'one']]
    index = MultiIndex.from_arrays(arrays, names=['first', 'second'])
    ser = Series(np.ones(5), index=index)
    result = ser.rename(index={'one': 'yes'}, level='second', errors='raise')
    arrays_expected = [['bar', 'baz', 'baz', 'foo', 'qux'], ['yes', 'yes', 'two', 'two', 'yes']]
    index_expected = MultiIndex.from_arrays(arrays_expected, names=['first', 'second'])
    series_expected = Series(np.ones(5), index=index_expected)
    tm.assert_series_equal(result, series_expected)