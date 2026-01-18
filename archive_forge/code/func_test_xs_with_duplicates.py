import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('key, level', [('one', 'second'), (['one'], ['second'])])
def test_xs_with_duplicates(self, key, level, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    df = concat([frame] * 2)
    assert df.index.is_unique is False
    expected = concat([frame.xs('one', level='second')] * 2)
    if isinstance(key, list):
        result = df.xs(tuple(key), level=level)
    else:
        result = df.xs(key, level=level)
    tm.assert_frame_equal(result, expected)