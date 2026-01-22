import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
class TestDatetimelikeCoercion:

    def test_setitem_dt64_string_scalar(self, tz_naive_fixture, indexer_sli):
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))
        values = ser._values
        newval = '2018-01-01'
        values._validate_setitem_value(newval)
        indexer_sli(ser)[0] = newval
        if tz is None:
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    def test_setitem_dt64_string_values(self, tz_naive_fixture, indexer_sli, key, box):
        tz = tz_naive_fixture
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)
        dti = date_range('2016-01-01', periods=3, tz=tz)
        ser = Series(dti.copy(deep=True))
        values = ser._values
        newvals = box(['2019-01-01', '2010-01-02'])
        values._validate_setitem_value(newvals)
        indexer_sli(ser)[key] = newvals
        if tz is None:
            assert ser.dtype == dti.dtype
            assert ser._values._ndarray is values._ndarray
        else:
            assert ser._values is values

    @pytest.mark.parametrize('scalar', ['3 Days', offsets.Hour(4)])
    def test_setitem_td64_scalar(self, indexer_sli, scalar):
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi.copy(deep=True))
        values = ser._values
        values._validate_setitem_value(scalar)
        indexer_sli(ser)[0] = scalar
        assert ser._values._ndarray is values._ndarray

    @pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
    @pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
    def test_setitem_td64_string_values(self, indexer_sli, key, box):
        if isinstance(key, slice) and indexer_sli is tm.loc:
            key = slice(0, 1)
        tdi = timedelta_range('1 Day', periods=3)
        ser = Series(tdi.copy(deep=True))
        values = ser._values
        newvals = box(['10 Days', '44 hours'])
        values._validate_setitem_value(newvals)
        indexer_sli(ser)[key] = newvals
        assert ser._values._ndarray is values._ndarray