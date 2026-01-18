from numpy import array, kron, diag
from numpy.testing import assert_, assert_equal
from scipy.sparse import _spfuncs as spfuncs
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix
from scipy.sparse._sparsetools import (csr_scale_rows, csr_scale_columns,
def test_scale_rows_and_cols(self):
    D = array([[1, 0, 0, 2, 3], [0, 4, 0, 5, 0], [0, 0, 6, 7, 0]])
    S = csr_matrix(D)
    v = array([1, 2, 3])
    csr_scale_rows(3, 5, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), diag(v) @ D)
    S = csr_matrix(D)
    v = array([1, 2, 3, 4, 5])
    csr_scale_columns(3, 5, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), D @ diag(v))
    E = kron(D, [[1, 2], [3, 4]])
    S = bsr_matrix(E, blocksize=(2, 2))
    v = array([1, 2, 3, 4, 5, 6])
    bsr_scale_rows(3, 5, 2, 2, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), diag(v) @ E)
    S = bsr_matrix(E, blocksize=(2, 2))
    v = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bsr_scale_columns(3, 5, 2, 2, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), E @ diag(v))
    E = kron(D, [[1, 2, 3], [4, 5, 6]])
    S = bsr_matrix(E, blocksize=(2, 3))
    v = array([1, 2, 3, 4, 5, 6])
    bsr_scale_rows(3, 5, 2, 3, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), diag(v) @ E)
    S = bsr_matrix(E, blocksize=(2, 3))
    v = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    bsr_scale_columns(3, 5, 2, 3, S.indptr, S.indices, S.data, v)
    assert_equal(S.toarray(), E @ diag(v))