from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
class TestEmptyDataFrameReductions:

    @pytest.mark.parametrize('opname, dtype, exp_value, exp_dtype', [('sum', np.int8, 0, np.int64), ('prod', np.int8, 1, np.int_), ('sum', np.int64, 0, np.int64), ('prod', np.int64, 1, np.int64), ('sum', np.uint8, 0, np.uint64), ('prod', np.uint8, 1, np.uint), ('sum', np.uint64, 0, np.uint64), ('prod', np.uint64, 1, np.uint64), ('sum', np.float32, 0, np.float32), ('prod', np.float32, 1, np.float32), ('sum', np.float64, 0, np.float64)])
    def test_df_empty_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)
        expected = Series([exp_value, exp_value], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('opname, dtype, exp_dtype', [('sum', np.int8, np.float64), ('prod', np.int8, np.float64), ('sum', np.int64, np.float64), ('prod', np.int64, np.float64), ('sum', np.uint8, np.float64), ('prod', np.uint8, np.float64), ('sum', np.uint64, np.float64), ('prod', np.uint64, np.float64), ('sum', np.float32, np.float32), ('prod', np.float32, np.float32), ('sum', np.float64, np.float64)])
    def test_df_empty_min_count_1(self, opname, dtype, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)
        expected = Series([np.nan, np.nan], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('opname, dtype, exp_value, exp_dtype', [('sum', 'Int8', 0, 'Int32' if is_windows_np2_or_is32 else 'Int64'), ('prod', 'Int8', 1, 'Int32' if is_windows_np2_or_is32 else 'Int64'), ('prod', 'Int8', 1, 'Int32' if is_windows_np2_or_is32 else 'Int64'), ('sum', 'Int64', 0, 'Int64'), ('prod', 'Int64', 1, 'Int64'), ('sum', 'UInt8', 0, 'UInt32' if is_windows_np2_or_is32 else 'UInt64'), ('prod', 'UInt8', 1, 'UInt32' if is_windows_np2_or_is32 else 'UInt64'), ('sum', 'UInt64', 0, 'UInt64'), ('prod', 'UInt64', 1, 'UInt64'), ('sum', 'Float32', 0, 'Float32'), ('prod', 'Float32', 1, 'Float32'), ('sum', 'Float64', 0, 'Float64')])
    def test_df_empty_nullable_min_count_0(self, opname, dtype, exp_value, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)
        expected = Series([exp_value, exp_value], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('opname, dtype, exp_dtype', [('sum', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'), ('prod', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'), ('sum', 'Int64', 'Int64'), ('prod', 'Int64', 'Int64'), ('sum', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'), ('prod', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'), ('sum', 'UInt64', 'UInt64'), ('prod', 'UInt64', 'UInt64'), ('sum', 'Float32', 'Float32'), ('prod', 'Float32', 'Float32'), ('sum', 'Float64', 'Float64')])
    def test_df_empty_nullable_min_count_1(self, opname, dtype, exp_dtype):
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)
        expected = Series([pd.NA, pd.NA], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)