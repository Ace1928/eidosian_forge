import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_values_multiindex_periodindex():
    ints = np.arange(2007, 2012)
    pidx = pd.PeriodIndex(ints, freq='D')
    idx = MultiIndex.from_arrays([ints, pidx])
    result = idx.values
    outer = Index([x[0] for x in result])
    tm.assert_index_equal(outer, Index(ints, dtype=np.int64))
    inner = pd.PeriodIndex([x[1] for x in result])
    tm.assert_index_equal(inner, pidx)
    result = idx[:2].values
    outer = Index([x[0] for x in result])
    tm.assert_index_equal(outer, Index(ints[:2], dtype=np.int64))
    inner = pd.PeriodIndex([x[1] for x in result])
    tm.assert_index_equal(inner, pidx[:2])