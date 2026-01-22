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
class TestAllowNonNano:

    @pytest.fixture(params=[True, False])
    def as_td(self, request):
        return request.param

    @pytest.fixture
    def arr(self, as_td):
        values = np.arange(5).astype(np.int64).view('M8[s]')
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    def test_index_allow_non_nano(self, arr):
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    def test_dti_tdi_allow_non_nano(self, arr, as_td):
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    def test_series_allow_non_nano(self, arr):
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    def test_frame_allow_non_nano(self, arr):
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    def test_frame_from_dict_allow_non_nano(self, arr):
        df = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype