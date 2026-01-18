from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes_scale_invariance():
    np.random.seed(1234)
    m, n = (4, 3)
    for i in range(3):
        A_orig = np.random.randn(m, n)
        B_orig = np.random.randn(m, n)
        R_orig, s = orthogonal_procrustes(A_orig, B_orig)
        for A_scale in np.square(np.random.randn(3)):
            for B_scale in np.square(np.random.randn(3)):
                R, s = orthogonal_procrustes(A_orig * A_scale, B_orig * B_scale)
                assert_allclose(R, R_orig)