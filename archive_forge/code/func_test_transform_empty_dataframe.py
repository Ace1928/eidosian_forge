import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import frame_transform_kernels
from pandas.tests.frame.common import zip_frames
def test_transform_empty_dataframe():
    df = DataFrame([], columns=['col1', 'col2'])
    result = df.transform(lambda x: x + 10)
    tm.assert_frame_equal(result, df)
    result = df['col1'].transform(lambda x: x + 10)
    tm.assert_series_equal(result, df['col1'])