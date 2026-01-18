import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth2():
    df = DataFrame({'color': {0: 'green', 1: 'green', 2: 'red', 3: 'red', 4: 'red'}, 'food': {0: 'ham', 1: 'eggs', 2: 'eggs', 3: 'ham', 4: 'pork'}, 'two': {0: 1.5456590000000001, 1: -0.070345, 2: -2.400454, 3: 0.46206, 4: 0.523508}, 'one': {0: 0.565738, 1: -0.9742360000000001, 2: 1.033801, 3: -0.785435, 4: 0.704228}}).set_index(['color', 'food'])
    result = df.groupby(level=0, as_index=False).nth(2)
    expected = df.iloc[[-1]]
    tm.assert_frame_equal(result, expected)
    result = df.groupby(level=0, as_index=False).nth(3)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)