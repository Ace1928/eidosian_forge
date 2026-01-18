import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
def test_sparse_vector_norms(self):
    for sparse_type in self._sparse_types:
        for M in self._test_matrices:
            S = sparse_type(M)
            for axis in (0, 1, -1, -2, (0,), (1,), (-1,), (-2,)):
                assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                for ord in (None, 2, np.inf, -np.inf, 1, 0.5, 0.42):
                    assert_allclose(spnorm(S, ord, axis=axis), npnorm(M, ord, axis=axis))