import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs', [{'dtype': 'float64'}, {'index': [1, 2, 4]}, {'name': 'foo'}])
def test_series_not_equal_metadata_mismatch(kwargs):
    data = range(3)
    s1 = Series(data)
    s2 = Series(data, **kwargs)
    _assert_not_series_equal_both(s1, s2)