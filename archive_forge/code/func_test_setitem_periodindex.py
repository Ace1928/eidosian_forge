from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_periodindex(self):
    rng = period_range('1/1/2000', periods=5, name='index')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=rng)
    df['Index'] = rng
    rs = Index(df['Index'])
    tm.assert_index_equal(rs, rng, check_names=False)
    assert rs.name == 'Index'
    assert rng.name == 'index'
    rs = df.reset_index().set_index('index')
    assert isinstance(rs.index, PeriodIndex)
    tm.assert_index_equal(rs.index, rng)