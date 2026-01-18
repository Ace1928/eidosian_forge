import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_wrap_agg_out(three_group):
    grouped = three_group.groupby(['A', 'B'])

    def func(ser):
        if ser.dtype == object:
            raise TypeError('Test error message')
        return ser.sum()
    with pytest.raises(TypeError, match='Test error message'):
        grouped.aggregate(func)
    result = grouped[['D', 'E', 'F']].aggregate(func)
    exp_grouped = three_group.loc[:, ['A', 'B', 'D', 'E', 'F']]
    expected = exp_grouped.groupby(['A', 'B']).aggregate(func)
    tm.assert_frame_equal(result, expected)