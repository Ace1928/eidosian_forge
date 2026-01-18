from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_records_no_typeerror_in_repr(self):
    df = DataFrame([['a', 'b'], ['c', 'd'], ['e', 'f']], columns=['left', 'right'])
    df['record'] = df[['left', 'right']].to_records()
    expected = '  left right     record\n0    a     b  [0, a, b]\n1    c     d  [1, c, d]\n2    e     f  [2, e, f]'
    result = repr(df)
    assert result == expected