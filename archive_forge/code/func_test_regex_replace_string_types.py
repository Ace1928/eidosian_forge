from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,to_replace,expected', [(['xax', 'xbx'], {'a': 'c', 'b': 'd'}, ['xcx', 'xdx']), (['d', '', ''], {'^\\s*$': pd.NA}, ['d', pd.NA, pd.NA])])
def test_regex_replace_string_types(self, data, to_replace, expected, frame_or_series, any_string_dtype, using_infer_string, request):
    dtype = any_string_dtype
    obj = frame_or_series(data, dtype=dtype)
    if using_infer_string and any_string_dtype == 'object':
        if len(to_replace) > 1 and isinstance(obj, DataFrame):
            request.node.add_marker(pytest.mark.xfail(reason='object input array that gets downcasted raises on second pass'))
        with tm.assert_produces_warning(FutureWarning, match='Downcasting'):
            result = obj.replace(to_replace, regex=True)
            dtype = 'string[pyarrow_numpy]'
    else:
        result = obj.replace(to_replace, regex=True)
    expected = frame_or_series(expected, dtype=dtype)
    tm.assert_equal(result, expected)