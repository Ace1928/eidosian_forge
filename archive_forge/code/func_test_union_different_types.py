from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_different_types(index_flat, index_flat2, request):
    idx1 = index_flat
    idx2 = index_flat2
    if not idx1.is_unique and (not idx2.is_unique) and (idx1.dtype.kind == 'i') and (idx2.dtype.kind == 'b') or (not idx2.is_unique and (not idx1.is_unique) and (idx2.dtype.kind == 'i') and (idx1.dtype.kind == 'b')):
        mark = pytest.mark.xfail(reason='GH#44000 True==1', raises=ValueError, strict=False)
        request.applymarker(mark)
    common_dtype = find_common_type([idx1.dtype, idx2.dtype])
    warn = None
    msg = "'<' not supported between"
    if not len(idx1) or not len(idx2):
        pass
    elif idx1.dtype.kind == 'c' and (not lib.is_np_dtype(idx2.dtype, 'iufc')) or (idx2.dtype.kind == 'c' and (not lib.is_np_dtype(idx1.dtype, 'iufc'))):
        warn = RuntimeWarning
    elif isinstance(idx1.dtype, PeriodDtype) and isinstance(idx2.dtype, CategoricalDtype) or (isinstance(idx2.dtype, PeriodDtype) and isinstance(idx1.dtype, CategoricalDtype)):
        warn = FutureWarning
        msg = 'PeriodDtype\\[B\\] is deprecated'
        mark = pytest.mark.xfail(reason='Warning not produced on all builds', raises=AssertionError, strict=False)
        request.applymarker(mark)
    any_uint64 = np.uint64 in (idx1.dtype, idx2.dtype)
    idx1_signed = is_signed_integer_dtype(idx1.dtype)
    idx2_signed = is_signed_integer_dtype(idx2.dtype)
    idx1 = idx1.sort_values()
    idx2 = idx2.sort_values()
    with tm.assert_produces_warning(warn, match=msg):
        res1 = idx1.union(idx2)
        res2 = idx2.union(idx1)
    if any_uint64 and (idx1_signed or idx2_signed):
        assert res1.dtype == np.dtype('O')
        assert res2.dtype == np.dtype('O')
    else:
        assert res1.dtype == common_dtype
        assert res2.dtype == common_dtype