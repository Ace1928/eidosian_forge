import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_values_with_duplicate_columns(self):
    df = DataFrame([[1, 2.5], [3, 4.5]], index=[1, 2], columns=['x', 'x'])
    result = df.values
    expected = np.array([[1, 2.5], [3, 4.5]])
    assert (result == expected).all().all()