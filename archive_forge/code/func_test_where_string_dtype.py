from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_string_dtype(frame_or_series):
    obj = frame_or_series(['a', 'b', 'c', 'd'], index=['id1', 'id2', 'id3', 'id4'], dtype=StringDtype())
    filtered_obj = frame_or_series(['b', 'c'], index=['id2', 'id3'], dtype=StringDtype())
    filter_ser = Series([False, True, True, False])
    result = obj.where(filter_ser, filtered_obj)
    expected = frame_or_series([pd.NA, 'b', 'c', pd.NA], index=['id1', 'id2', 'id3', 'id4'], dtype=StringDtype())
    tm.assert_equal(result, expected)
    result = obj.mask(~filter_ser, filtered_obj)
    tm.assert_equal(result, expected)
    obj.mask(~filter_ser, filtered_obj, inplace=True)
    tm.assert_equal(result, expected)