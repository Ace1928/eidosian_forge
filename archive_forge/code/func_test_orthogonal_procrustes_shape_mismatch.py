from itertools import product, permutations
import numpy as np
from numpy.testing import assert_array_less, assert_allclose
from pytest import raises as assert_raises
from scipy.linalg import inv, eigh, norm
from scipy.linalg import orthogonal_procrustes
from scipy.sparse._sputils import matrix
def test_orthogonal_procrustes_shape_mismatch():
    np.random.seed(1234)
    shapes = ((3, 3), (3, 4), (4, 3), (4, 4))
    for a, b in permutations(shapes, 2):
        A = np.random.randn(*a)
        B = np.random.randn(*b)
        assert_raises(ValueError, orthogonal_procrustes, A, B)