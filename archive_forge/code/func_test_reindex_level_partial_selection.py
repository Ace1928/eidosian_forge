import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_level_partial_selection(self, multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data
    result = frame.reindex(['foo', 'qux'], level=0)
    expected = frame.iloc[[0, 1, 2, 7, 8, 9]]
    tm.assert_frame_equal(result, expected)
    result = frame.T.reindex(['foo', 'qux'], axis=1, level=0)
    tm.assert_frame_equal(result, expected.T)
    result = frame.loc[['foo', 'qux']]
    tm.assert_frame_equal(result, expected)
    result = frame['A'].loc[['foo', 'qux']]
    tm.assert_series_equal(result, expected['A'])
    result = frame.T.loc[:, ['foo', 'qux']]
    tm.assert_frame_equal(result, expected.T)