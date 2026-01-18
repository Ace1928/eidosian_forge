from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_single_column_ea_index_and_columns(self, any_numeric_ea_dtype):
    df = DataFrame({'a': [1, 2]}, dtype=any_numeric_ea_dtype)
    result = df.reindex(columns=list('ab'), index=[0, 1, 2], fill_value=10)
    expected = DataFrame({'a': Series([1, 2, 10], dtype=any_numeric_ea_dtype), 'b': 10})
    tm.assert_frame_equal(result, expected)