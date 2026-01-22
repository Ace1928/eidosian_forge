import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def _check_bool_result(self, res):
    assert isinstance(res, SparseArray)
    assert isinstance(res.dtype, SparseDtype)
    assert res.dtype.subtype == np.bool_
    assert isinstance(res.fill_value, bool)