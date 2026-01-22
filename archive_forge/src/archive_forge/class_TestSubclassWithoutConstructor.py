import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestSubclassWithoutConstructor:

    def test_copy_df(self):
        expected = DataFrame({'a': [1, 2, 3]})
        result = SimpleDataFrameSubClass(expected).copy()
        assert type(result) is DataFrame
        tm.assert_frame_equal(result, expected)

    def test_copy_series(self):
        expected = Series([1, 2, 3])
        result = SimpleSeriesSubClass(expected).copy()
        tm.assert_series_equal(result, expected)

    def test_series_to_frame(self):
        orig = Series([1, 2, 3])
        expected = orig.to_frame()
        result = SimpleSeriesSubClass(orig).to_frame()
        assert type(result) is DataFrame
        tm.assert_frame_equal(result, expected)

    def test_groupby(self):
        df = SimpleDataFrameSubClass(DataFrame({'a': [1, 2, 3]}))
        for _, v in df.groupby('a'):
            assert type(v) is DataFrame