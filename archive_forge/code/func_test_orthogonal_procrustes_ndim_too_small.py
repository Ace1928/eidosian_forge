from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes_ndim_too_small():
    np.random.seed(1234)
    A = np.random.randn(3)
    B = np.random.randn(3)
    assert_raises(ValueError, orthogonal_procrustes, A, B)