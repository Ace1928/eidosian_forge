import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('kwargs,msg', [({'duplicates': 'drop'}, None), ({}, 'Bin edges must be unique'), ({'duplicates': 'raise'}, 'Bin edges must be unique'), ({'duplicates': 'foo'}, "invalid value for 'duplicates' parameter")])
def test_qcut_duplicates_bin(kwargs, msg):
    values = [0, 0, 0, 0, 1, 2, 3]
    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            qcut(values, 3, **kwargs)
    else:
        result = qcut(values, 3, **kwargs)
        expected = IntervalIndex([Interval(-0.001, 1), Interval(1, 3)])
        tm.assert_index_equal(result.categories, expected)