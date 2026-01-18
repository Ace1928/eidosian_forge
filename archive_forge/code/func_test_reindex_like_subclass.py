import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_reindex_like_subclass(self):

    class MyDataFrame(DataFrame):
        pass
    expected = DataFrame()
    df = MyDataFrame()
    result = df.reindex_like(expected)
    tm.assert_frame_equal(result, expected)