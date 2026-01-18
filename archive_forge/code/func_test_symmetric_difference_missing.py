from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('index2,expected', [(Index([0, 1, np.nan]), Index([2.0, 3.0, 0.0])), (Index([0, 1]), Index([np.nan, 2.0, 3.0, 0.0]))])
def test_symmetric_difference_missing(self, index2, expected, sort):
    index1 = Index([1, np.nan, 2, 3])
    result = index1.symmetric_difference(index2, sort=sort)
    if sort is None:
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)