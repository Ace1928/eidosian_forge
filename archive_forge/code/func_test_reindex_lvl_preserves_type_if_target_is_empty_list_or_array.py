import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reindex_lvl_preserves_type_if_target_is_empty_list_or_array(using_infer_string):
    idx = MultiIndex.from_product([[0, 1], ['a', 'b']])
    assert idx.reindex([], level=0)[0].levels[0].dtype.type == np.int64
    exp = np.object_ if not using_infer_string else str
    assert idx.reindex([], level=1)[0].levels[1].dtype.type == exp
    cat = pd.Categorical(['foo', 'bar'])
    dti = pd.date_range('2016-01-01', periods=2, tz='US/Pacific')
    mi = MultiIndex.from_product([cat, dti])
    assert mi.reindex([], level=0)[0].levels[0].dtype == cat.dtype
    assert mi.reindex([], level=1)[0].levels[1].dtype == dti.dtype