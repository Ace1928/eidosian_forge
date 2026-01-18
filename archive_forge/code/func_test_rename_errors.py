from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('mapper, errors, expected_columns', [({'A': 'a', 'E': 'e'}, 'ignore', ['a', 'B', 'C', 'D']), ({'A': 'a'}, 'raise', ['a', 'B', 'C', 'D']), (str.lower, 'raise', ['a', 'b', 'c', 'd'])])
def test_rename_errors(self, mapper, errors, expected_columns):
    df = DataFrame(columns=['A', 'B', 'C', 'D'])
    result = df.rename(columns=mapper, errors=errors)
    expected = DataFrame(columns=expected_columns)
    tm.assert_frame_equal(result, expected)