import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_replacer_equals_replacement(self):
    s = pd.Series(['a', 'b'])
    expected = pd.Series(['b', 'a'])
    result = s.replace({'a': 'b', 'b': 'a'})
    tm.assert_series_equal(expected, result)