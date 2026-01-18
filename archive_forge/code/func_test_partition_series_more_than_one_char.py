from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, exp', [['partition', [('a', '__', 'b__c'), ('c', '__', 'd__e'), np.nan, ('f', '__', 'g__h'), None]], ['rpartition', [('a__b', '__', 'c'), ('c__d', '__', 'e'), np.nan, ('f__g', '__', 'h'), None]]])
def test_partition_series_more_than_one_char(method, exp, any_string_dtype):
    s = Series(['a__b__c', 'c__d__e', np.nan, 'f__g__h', None], dtype=any_string_dtype)
    result = getattr(s.str, method)('__', expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)