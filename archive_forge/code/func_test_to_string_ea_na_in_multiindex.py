from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_to_string_ea_na_in_multiindex(self):
    df = DataFrame({'a': [1, 2]}, index=MultiIndex.from_arrays([Series([NA, 1], dtype='Int64')]))
    result = df.to_string()
    expected = '      a\n<NA>  1\n1     2'
    assert result == expected