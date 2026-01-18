import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_permutation(self):
    A = block_diag(np.ones((2, 2)), np.tril(np.ones((2, 2))), np.ones((3, 3)))
    x, (y, z) = matrix_balance(A, separate=1)
    assert_allclose(y, np.ones_like(y))
    assert_allclose(z, np.array([0, 1, 6, 5, 4, 3, 2]))