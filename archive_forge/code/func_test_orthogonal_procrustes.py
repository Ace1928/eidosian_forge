from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes():
    np.random.seed(1234)
    for m, n in ((6, 4), (4, 4), (4, 6)):
        B = np.random.randn(m, n)
        X = np.random.randn(n, n)
        w, V = eigh(X.T + X)
        assert_allclose(inv(V), V.T)
        A = np.dot(B, V.T)
        R, s = orthogonal_procrustes(A, B)
        assert_allclose(inv(R), R.T)
        assert_allclose(A.dot(R), B)
        A_perturbed = A + 0.01 * np.random.randn(m, n)
        R_prime, s = orthogonal_procrustes(A_perturbed, B)
        assert_allclose(inv(R_prime), R_prime.T)
        naive_approx = A_perturbed.dot(R)
        optim_approx = A_perturbed.dot(R_prime)
        naive_approx_error = norm(naive_approx - B, ord='fro')
        optim_approx_error = norm(optim_approx - B, ord='fro')
        assert_array_less(optim_approx_error, naive_approx_error)