from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_with_name_series(any_string_dtype):
    s = Series(['a,b', 'c,d'], name='xxx', dtype=any_string_dtype)
    res = s.str.split(',')
    exp = Series([['a', 'b'], ['c', 'd']], name='xxx')
    tm.assert_series_equal(res, exp)
    res = s.str.split(',', expand=True)
    exp = DataFrame([['a', 'b'], ['c', 'd']], dtype=any_string_dtype)
    tm.assert_frame_equal(res, exp)