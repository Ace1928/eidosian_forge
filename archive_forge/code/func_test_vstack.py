import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
@pytest.mark.parametrize('coo_cls', [coo_matrix, coo_array])
def test_vstack(self, coo_cls):
    A = coo_cls([[1, 2], [3, 4]])
    B = coo_cls([[5, 6]])
    expected = array([[1, 2], [3, 4], [5, 6]])
    assert_equal(construct.vstack([A, B]).toarray(), expected)
    assert_equal(construct.vstack([A, B], dtype=np.float32).dtype, np.float32)
    assert_equal(construct.vstack([A.tocsr(), B.tocsr()]).toarray(), expected)
    result = construct.vstack([A.tocsr(), B.tocsr()], format='csr', dtype=np.float32)
    assert_equal(result.dtype, np.float32)
    assert_equal(result.indices.dtype, np.int32)
    assert_equal(result.indptr.dtype, np.int32)
    assert_equal(construct.vstack([A.tocsc(), B.tocsc()]).toarray(), expected)
    result = construct.vstack([A.tocsc(), B.tocsc()], format='csc', dtype=np.float32)
    assert_equal(result.dtype, np.float32)
    assert_equal(result.indices.dtype, np.int32)
    assert_equal(result.indptr.dtype, np.int32)