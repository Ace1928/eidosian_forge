from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('data, pat, expected', [(['split once', 'split once too!'], None, Series({0: ['split', 'once'], 1: ['split', 'once too!']})), (['split_once', 'split_once_too!'], '_', Series({0: ['split', 'once'], 1: ['split', 'once_too!']}))])
def test_split_no_pat_with_nonzero_n(data, pat, expected, any_string_dtype):
    s = Series(data, dtype=any_string_dtype)
    result = s.str.split(pat=pat, n=1)
    tm.assert_series_equal(expected, result, check_index_type=False)