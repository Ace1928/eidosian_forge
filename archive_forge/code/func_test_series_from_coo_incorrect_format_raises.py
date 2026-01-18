import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_series_from_coo_incorrect_format_raises(self):
    sp_sparse = pytest.importorskip('scipy.sparse')
    m = sp_sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
    with pytest.raises(TypeError, match='Expected coo_matrix. Got csr_matrix instead.'):
        pd.Series.sparse.from_coo(m)