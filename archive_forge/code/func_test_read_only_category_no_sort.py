from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_read_only_category_no_sort():
    cats = np.array([1, 2])
    cats.flags.writeable = False
    df = DataFrame({'a': [1, 3, 5, 7], 'b': Categorical([1, 1, 2, 2], categories=Index(cats))})
    expected = DataFrame(data={'a': [2.0, 6.0]}, index=CategoricalIndex(cats, name='b'))
    result = df.groupby('b', sort=False, observed=False).mean()
    tm.assert_frame_equal(result, expected)