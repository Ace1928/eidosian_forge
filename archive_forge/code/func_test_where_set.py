from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.filterwarnings('ignore:Downcasting object dtype arrays:FutureWarning')
def test_where_set(self, where_frame, float_string_frame, mixed_int_frame):

    def _check_set(df, cond, check_dtypes=True):
        dfi = df.copy()
        econd = cond.reindex_like(df).fillna(True).infer_objects(copy=False)
        expected = dfi.mask(~econd)
        return_value = dfi.where(cond, np.nan, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(dfi, expected)
        if check_dtypes:
            for k, v in df.dtypes.items():
                if issubclass(v.type, np.integer) and (not cond[k].all()):
                    v = np.dtype('float64')
                assert dfi[k].dtype == v
    df = where_frame
    if df is float_string_frame:
        msg = "'>' not supported between instances of 'str' and 'int'"
        with pytest.raises(TypeError, match=msg):
            df > 0
        return
    if df is mixed_int_frame:
        df = df.astype('float64')
    cond = df > 0
    _check_set(df, cond)
    cond = df >= 0
    _check_set(df, cond)
    cond = (df >= 0)[1:]
    _check_set(df, cond)