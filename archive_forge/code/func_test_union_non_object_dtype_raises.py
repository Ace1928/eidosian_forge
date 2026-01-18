import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_non_object_dtype_raises():
    mi = MultiIndex.from_product([['a', 'b'], [1, 2]])
    idx = mi.levels[1]
    msg = 'Can only union MultiIndex with MultiIndex or Index of tuples'
    with pytest.raises(NotImplementedError, match=msg):
        mi.union(idx)