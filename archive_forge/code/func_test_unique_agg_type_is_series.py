from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('test, constant', [({'a': [1, 2, 3], 'b': [1, 1, 1]}, {'a': [1, 2, 3], 'b': [1]}), ({'a': [2, 2, 2], 'b': [1, 1, 1]}, {'a': [2], 'b': [1]})])
def test_unique_agg_type_is_series(test, constant):
    df1 = DataFrame(test)
    expected = Series(data=constant, index=['a', 'b'], dtype='object')
    aggregation = {'a': 'unique', 'b': 'unique'}
    result = df1.agg(aggregation)
    tm.assert_series_equal(result, expected)