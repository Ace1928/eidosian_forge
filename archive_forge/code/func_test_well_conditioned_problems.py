import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
def test_well_conditioned_problems():
    n = 10
    A_sparse = scipy.sparse.eye(n, n)
    A_dense = A_sparse.toarray()
    with np.errstate(invalid='raise'):
        for seed in range(30):
            rng = np.random.RandomState(seed + 10)
            beta = rng.rand(n)
            beta[beta == 0] = 1e-05
            b = A_sparse @ beta[:, np.newaxis]
            output = lsqr(A_sparse, b, show=show)
            assert_equal(output[1], 1)
            solution = output[0]
            assert_allclose(solution, beta)
            reference_solution = np.linalg.solve(A_dense, b).ravel()
            assert_allclose(solution, reference_solution)