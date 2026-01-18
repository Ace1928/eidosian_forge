from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_str_accessor_in_apply_func():
    df = DataFrame(zip('abc', 'def'))
    expected = Series(['A/D', 'B/E', 'C/F'])
    result = df.apply(lambda f: '/'.join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)