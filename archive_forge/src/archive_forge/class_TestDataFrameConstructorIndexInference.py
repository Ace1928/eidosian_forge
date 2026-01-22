import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
class TestDataFrameConstructorIndexInference:

    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self):
        rng1 = pd.period_range('1/1/1999', '1/1/2012', freq='M')
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)
        rng2 = pd.period_range('1/1/1980', '12/1/2001', freq='M')
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({'s1': s1, 's2': s2})
        exp = pd.period_range('1/1/1980', '1/1/2012', freq='M')
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self):
        dti = date_range('2016-01-01', periods=3)
        ser1 = Series(range(3), index=dti)
        ser2 = Series(range(3), index=dti.tz_localize('UTC'))
        ser3 = Series(range(3), index=dti.tz_localize('US/Central'))
        ser4 = Series(range(3))
        df1 = DataFrame({'A': ser2, 'B': ser3, 'C': ser4})
        exp_index = Index(list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object)
        tm.assert_index_equal(df1.index, exp_index)
        df2 = DataFrame({'A': ser2, 'C': ser4, 'B': ser3})
        exp_index3 = Index(list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object)
        tm.assert_index_equal(df2.index, exp_index3)
        df3 = DataFrame({'B': ser3, 'A': ser2, 'C': ser4})
        exp_index3 = Index(list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object)
        tm.assert_index_equal(df3.index, exp_index3)
        df4 = DataFrame({'C': ser4, 'B': ser3, 'A': ser2})
        exp_index4 = Index(list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object)
        tm.assert_index_equal(df4.index, exp_index4)
        msg = 'Cannot join tz-naive with tz-aware DatetimeIndex'
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'C': ser4, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'A': ser2, 'B': ser3, 'D': ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({'D': ser1, 'A': ser2, 'B': ser3})

    @pytest.mark.parametrize('key_val, col_vals, col_type', [['3', ['3', '4'], 'utf8'], [3, [3, 4], 'int8']])
    def test_dict_data_arrow_column_expansion(self, key_val, col_vals, col_type):
        pa = pytest.importorskip('pyarrow')
        cols = pd.arrays.ArrowExtensionArray(pa.array(col_vals, type=pa.dictionary(pa.int8(), getattr(pa, col_type)())))
        result = DataFrame({key_val: [1, 2]}, columns=cols)
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        tm.assert_frame_equal(result, expected)