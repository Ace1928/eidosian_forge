import operator
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def _check_comparison_ops(self, a, b, a_dense, b_dense):
    with np.errstate(invalid='ignore'):
        self._check_bool_result(a == b)
        self._assert((a == b).to_dense(), a_dense == b_dense)
        self._check_bool_result(a != b)
        self._assert((a != b).to_dense(), a_dense != b_dense)
        self._check_bool_result(a >= b)
        self._assert((a >= b).to_dense(), a_dense >= b_dense)
        self._check_bool_result(a <= b)
        self._assert((a <= b).to_dense(), a_dense <= b_dense)
        self._check_bool_result(a > b)
        self._assert((a > b).to_dense(), a_dense > b_dense)
        self._check_bool_result(a < b)
        self._assert((a < b).to_dense(), a_dense < b_dense)
        self._check_bool_result(a == b_dense)
        self._assert((a == b_dense).to_dense(), a_dense == b_dense)
        self._check_bool_result(a != b_dense)
        self._assert((a != b_dense).to_dense(), a_dense != b_dense)
        self._check_bool_result(a >= b_dense)
        self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)
        self._check_bool_result(a <= b_dense)
        self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)
        self._check_bool_result(a > b_dense)
        self._assert((a > b_dense).to_dense(), a_dense > b_dense)
        self._check_bool_result(a < b_dense)
        self._assert((a < b_dense).to_dense(), a_dense < b_dense)