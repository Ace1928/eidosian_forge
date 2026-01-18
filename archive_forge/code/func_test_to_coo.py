import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('colnames', [('A', 'B'), (1, 2), (1, pd.NA), (0.1, 0.2), ('x', 'x'), (0, 0)])
def test_to_coo(self, colnames):
    sp_sparse = pytest.importorskip('scipy.sparse')
    df = pd.DataFrame({colnames[0]: [0, 1, 0], colnames[1]: [1, 0, 0]}, dtype='Sparse[int64, 0]')
    result = df.sparse.to_coo()
    expected = sp_sparse.coo_matrix(np.asarray(df))
    assert (result != expected).nnz == 0