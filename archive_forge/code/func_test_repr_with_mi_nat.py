from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_with_mi_nat(self):
    df = DataFrame({'X': [1, 2]}, index=[[NaT, Timestamp('20130101')], ['a', 'b']])
    result = repr(df)
    expected = '              X\nNaT        a  1\n2013-01-01 b  2'
    assert result == expected