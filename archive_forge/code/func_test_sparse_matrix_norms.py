import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
def test_sparse_matrix_norms(self):
    for sparse_type in self._sparse_types:
        for M in self._test_matrices:
            S = sparse_type(M)
            assert_allclose(spnorm(S), npnorm(M))
            assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
            assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
            assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
            assert_allclose(spnorm(S, 1), npnorm(M, 1))
            assert_allclose(spnorm(S, -1), npnorm(M, -1))