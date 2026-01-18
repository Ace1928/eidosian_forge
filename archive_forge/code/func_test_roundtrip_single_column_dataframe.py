import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_roundtrip_single_column_dataframe():
    categories = DataFrame({'': ['a', 'b', 'c', 'a']})
    dummies = get_dummies(categories)
    result = from_dummies(dummies, sep='_')
    expected = categories
    tm.assert_frame_equal(result, expected)