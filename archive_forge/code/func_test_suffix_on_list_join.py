from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_suffix_on_list_join():
    first = DataFrame({'key': [1, 2, 3, 4, 5]})
    second = DataFrame({'key': [1, 8, 3, 2, 5], 'v1': [1, 2, 3, 4, 5]})
    third = DataFrame({'keys': [5, 2, 3, 4, 1], 'v2': [1, 2, 3, 4, 5]})
    msg = 'Suffixes not supported when joining multiple DataFrames'
    with pytest.raises(ValueError, match=msg):
        first.join([second], lsuffix='y')
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], rsuffix='x')
    with pytest.raises(ValueError, match=msg):
        first.join([second, third], lsuffix='y', rsuffix='x')
    with pytest.raises(ValueError, match='Indexes have overlapping values'):
        first.join([second, third])
    arr_joined = first.join([third])
    norm_joined = first.join(third)
    tm.assert_frame_equal(arr_joined, norm_joined)