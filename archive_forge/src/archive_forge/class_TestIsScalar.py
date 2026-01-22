import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestIsScalar:

    def test_is_scalar_builtin_scalars(self):
        assert is_scalar(None)
        assert is_scalar(True)
        assert is_scalar(False)
        assert is_scalar(Fraction())
        assert is_scalar(0.0)
        assert is_scalar(1)
        assert is_scalar(complex(2))
        assert is_scalar(float('NaN'))
        assert is_scalar(np.nan)
        assert is_scalar('foobar')
        assert is_scalar(b'foobar')
        assert is_scalar(datetime(2014, 1, 1))
        assert is_scalar(date(2014, 1, 1))
        assert is_scalar(time(12, 0))
        assert is_scalar(timedelta(hours=1))
        assert is_scalar(pd.NaT)
        assert is_scalar(pd.NA)

    def test_is_scalar_builtin_nonscalars(self):
        assert not is_scalar({})
        assert not is_scalar([])
        assert not is_scalar([1])
        assert not is_scalar(())
        assert not is_scalar((1,))
        assert not is_scalar(slice(None))
        assert not is_scalar(Ellipsis)

    def test_is_scalar_numpy_array_scalars(self):
        assert is_scalar(np.int64(1))
        assert is_scalar(np.float64(1.0))
        assert is_scalar(np.int32(1))
        assert is_scalar(np.complex64(2))
        assert is_scalar(np.object_('foobar'))
        assert is_scalar(np.str_('foobar'))
        assert is_scalar(np.bytes_(b'foobar'))
        assert is_scalar(np.datetime64('2014-01-01'))
        assert is_scalar(np.timedelta64(1, 'h'))

    @pytest.mark.parametrize('zerodim', [np.array(1), np.array('foobar'), np.array(np.datetime64('2014-01-01')), np.array(np.timedelta64(1, 'h')), np.array(np.datetime64('NaT'))])
    def test_is_scalar_numpy_zerodim_arrays(self, zerodim):
        assert not is_scalar(zerodim)
        assert is_scalar(lib.item_from_zerodim(zerodim))

    @pytest.mark.parametrize('arr', [np.array([]), np.array([[]])])
    def test_is_scalar_numpy_arrays(self, arr):
        assert not is_scalar(arr)
        assert not is_scalar(MockNumpyLikeArray(arr))

    def test_is_scalar_pandas_scalars(self):
        assert is_scalar(Timestamp('2014-01-01'))
        assert is_scalar(Timedelta(hours=1))
        assert is_scalar(Period('2014-01-01'))
        assert is_scalar(Interval(left=0, right=1))
        assert is_scalar(DateOffset(days=1))
        assert is_scalar(pd.offsets.Minute(3))

    def test_is_scalar_pandas_containers(self):
        assert not is_scalar(Series(dtype=object))
        assert not is_scalar(Series([1]))
        assert not is_scalar(DataFrame())
        assert not is_scalar(DataFrame([[1]]))
        assert not is_scalar(Index([]))
        assert not is_scalar(Index([1]))
        assert not is_scalar(Categorical([]))
        assert not is_scalar(DatetimeIndex([])._data)
        assert not is_scalar(TimedeltaIndex([])._data)
        assert not is_scalar(DatetimeIndex([])._data.to_period('D'))
        assert not is_scalar(pd.array([1, 2, 3]))

    def test_is_scalar_number(self):

        class Numeric(Number):

            def __init__(self, value) -> None:
                self.value = value

            def __int__(self) -> int:
                return self.value
        num = Numeric(1)
        assert is_scalar(num)