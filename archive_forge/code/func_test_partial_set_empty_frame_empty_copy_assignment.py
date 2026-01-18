import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame_empty_copy_assignment(self):
    df = DataFrame(index=[0])
    df = df.copy()
    df['a'] = 0
    expected = DataFrame(0, index=[0], columns=Index(['a'], dtype=object))
    tm.assert_frame_equal(df, expected)