from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes_skbio_example():
    A_orig = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], dtype=float)
    B_orig = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], dtype=float)
    B_standardized = np.array([[-0.13363062, 0.6681531], [-0.13363062, 0.13363062], [-0.13363062, -0.40089186], [0.40089186, -0.40089186]])
    A, A_mu = _centered(A_orig)
    B, B_mu = _centered(B_orig)
    R, s = orthogonal_procrustes(A, B)
    scale = s / np.square(norm(A))
    B_approx = scale * np.dot(A, R) + B_mu
    assert_allclose(B_approx, B_orig)
    assert_allclose(B / norm(B), B_standardized)