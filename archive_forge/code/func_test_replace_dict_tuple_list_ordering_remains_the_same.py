from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_dict_tuple_list_ordering_remains_the_same(self):
    df = DataFrame({'A': [np.nan, 1]})
    res1 = df.replace(to_replace={np.nan: 0, 1: -100000000.0})
    res2 = df.replace(to_replace=(1, np.nan), value=[-100000000.0, 0])
    res3 = df.replace(to_replace=[1, np.nan], value=[-100000000.0, 0])
    expected = DataFrame({'A': [0, -100000000.0]})
    tm.assert_frame_equal(res1, res2)
    tm.assert_frame_equal(res2, res3)
    tm.assert_frame_equal(res3, expected)