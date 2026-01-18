import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_level_by_name(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    frame.index.names = ['first', 'second']
    result = frame.sort_index(level='second')
    expected = frame.sort_index(level=1)
    tm.assert_frame_equal(result, expected)