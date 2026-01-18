import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_update_dtypes(self):
    df = DataFrame([[1.0, 2.0, 1, False, True], [4.0, 5.0, 2, True, False]], columns=['A', 'B', 'int', 'bool1', 'bool2'])
    other = DataFrame([[45, 45, 3, True]], index=[0], columns=['A', 'B', 'int', 'bool1'])
    df.update(other)
    expected = DataFrame([[45.0, 45.0, 3, True, True], [4.0, 5.0, 2, True, False]], columns=['A', 'B', 'int', 'bool1', 'bool2'])
    tm.assert_frame_equal(df, expected)