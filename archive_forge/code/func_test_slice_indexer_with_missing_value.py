from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_arr,expected,start_idx,end_idx', [([[np.nan, 1, 2], [3, 4, 5]], slice(0, 2, None), np.nan, 1), ([[np.nan, 1, 2], [3, 4, 5]], slice(0, 3, None), np.nan, (2, 5)), ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), 3), ([[1, 2, 3], [4, np.nan, 5]], slice(1, 3, None), (2, np.nan), (3, 5))])
def test_slice_indexer_with_missing_value(index_arr, expected, start_idx, end_idx):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.slice_indexer(start=start_idx, end=end_idx)
    assert result == expected