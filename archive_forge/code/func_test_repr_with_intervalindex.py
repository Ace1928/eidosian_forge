from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_with_intervalindex(self):
    df = DataFrame({'A': [1, 2, 3, 4]}, index=IntervalIndex.from_breaks([0, 1, 2, 3, 4]))
    result = repr(df)
    expected = '        A\n(0, 1]  1\n(1, 2]  2\n(2, 3]  3\n(3, 4]  4'
    assert result == expected