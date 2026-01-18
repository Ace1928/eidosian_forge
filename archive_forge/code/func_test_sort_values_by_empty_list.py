import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_by_empty_list(self):
    expected = DataFrame({'a': [1, 4, 2, 5, 3, 6]})
    result = expected.sort_values(by=[])
    tm.assert_frame_equal(result, expected)
    assert result is not expected