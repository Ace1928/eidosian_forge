import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
    if isinstance(a_dense, np.ndarray):
        expected = op(pd.Series(a_dense), b_dense).values
    elif isinstance(b_dense, np.ndarray):
        expected = op(a_dense, pd.Series(b_dense)).values
    else:
        raise NotImplementedError
    with np.errstate(invalid='ignore', divide='ignore'):
        if mix:
            result = op(a, b_dense).to_dense()
        else:
            result = op(a, b).to_dense()
    self._assert(result, expected)