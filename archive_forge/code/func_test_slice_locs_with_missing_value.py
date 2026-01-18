from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_arr,expected,start_idx,end_idx', [([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, None), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, 'b'), ([[np.nan, 'a', 'b'], ['c', 'd', 'e']], (0, 3), np.nan, ('b', 'e')), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), None), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), 'c'), ([['a', 'b', 'c'], ['d', np.nan, 'e']], (1, 3), ('b', np.nan), ('c', 'e'))])
def test_slice_locs_with_missing_value(self, index_arr, expected, start_idx, end_idx):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.slice_locs(start=start_idx, end=end_idx)
    assert result == expected