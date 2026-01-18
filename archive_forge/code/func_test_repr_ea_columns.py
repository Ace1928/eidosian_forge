from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_ea_columns(self, any_string_dtype):
    pytest.importorskip('pyarrow')
    df = DataFrame({'long_column_name': [1, 2, 3], 'col2': [4, 5, 6]})
    df.columns = df.columns.astype(any_string_dtype)
    expected = '   long_column_name  col2\n0                 1     4\n1                 2     5\n2                 3     6'
    assert repr(df) == expected