import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_axis1(self):
    easy_weight_list = [0] * 3
    easy_weight_list[2] = 1
    df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': ['a'] * 10})
    sample1 = df.sample(n=1, axis=1, weights=easy_weight_list)
    tm.assert_frame_equal(sample1, df[['colString']])
    tm.assert_frame_equal(df.sample(n=3, random_state=42), df.sample(n=3, axis=0, random_state=42))