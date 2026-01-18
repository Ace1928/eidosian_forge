import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_row_norms(dtype, csr_container):
    X = np.random.RandomState(42).randn(100, 100)
    if dtype is np.float32:
        precision = 4
    else:
        precision = 5
    X = X.astype(dtype, copy=False)
    sq_norm = (X ** 2).sum(axis=1)
    assert_array_almost_equal(sq_norm, row_norms(X, squared=True), precision)
    assert_array_almost_equal(np.sqrt(sq_norm), row_norms(X), precision)
    for csr_index_dtype in [np.int32, np.int64]:
        Xcsr = csr_container(X, dtype=dtype)
        if csr_index_dtype is np.int64:
            Xcsr.indptr = Xcsr.indptr.astype(csr_index_dtype, copy=False)
            Xcsr.indices = Xcsr.indices.astype(csr_index_dtype, copy=False)
        assert Xcsr.indices.dtype == csr_index_dtype
        assert Xcsr.indptr.dtype == csr_index_dtype
        assert_array_almost_equal(sq_norm, row_norms(Xcsr, squared=True), precision)
        assert_array_almost_equal(np.sqrt(sq_norm), row_norms(Xcsr), precision)