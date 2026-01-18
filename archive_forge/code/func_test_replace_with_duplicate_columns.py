from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('replacement', [np.nan, 5])
def test_replace_with_duplicate_columns(self, replacement):
    result = DataFrame({'A': [1, 2, 3], 'A1': [4, 5, 6], 'B': [7, 8, 9]})
    result.columns = list('AAB')
    expected = DataFrame({'A': [1, 2, 3], 'A1': [4, 5, 6], 'B': [replacement, 8, 9]})
    expected.columns = list('AAB')
    result['B'] = result['B'].replace(7, replacement)
    tm.assert_frame_equal(result, expected)