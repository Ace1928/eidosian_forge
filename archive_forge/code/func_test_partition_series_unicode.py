from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, exp', [['partition', [('a', '_', 'b_c'), ('c', '_', 'd_e'), np.nan, ('f', '_', 'g_h')]], ['rpartition', [('a_b', '_', 'c'), ('c_d', '_', 'e'), np.nan, ('f_g', '_', 'h')]]])
def test_partition_series_unicode(any_string_dtype, method, exp):
    s = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype=any_string_dtype)
    result = getattr(s.str, method)('_', expand=False)
    expected = Series(exp)
    expected = _convert_na_value(s, expected)
    tm.assert_series_equal(result, expected)