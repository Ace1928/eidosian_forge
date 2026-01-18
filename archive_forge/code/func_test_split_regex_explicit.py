from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_split_regex_explicit(any_string_dtype):
    regex_pat = re.compile('.jpg')
    values = Series('xxxjpgzzz.jpg', dtype=any_string_dtype)
    result = values.str.split(regex_pat)
    exp = Series([['xx', 'zzz', '']])
    tm.assert_series_equal(result, exp)
    result = values.str.split('\\.jpg', regex=False)
    exp = Series([['xxxjpgzzz.jpg']])
    tm.assert_series_equal(result, exp)
    result = values.str.split('.')
    exp = Series([['xxxjpgzzz', 'jpg']])
    tm.assert_series_equal(result, exp)
    result = values.str.split('.jpg')
    exp = Series([['xx', 'zzz', '']])
    tm.assert_series_equal(result, exp)
    with pytest.raises(ValueError, match='Cannot use a compiled regex as replacement pattern with regex=False'):
        values.str.split(regex_pat, regex=False)