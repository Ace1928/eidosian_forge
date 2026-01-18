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
def test_transposed_keyword(self):
    A = np.arange(9).reshape(3, 3) + 1
    x = solve(np.tril(A) / 9, np.ones(3), transposed=True)
    assert_array_almost_equal(x, [1.2, 0.2, 1])
    x = solve(np.tril(A) / 9, np.ones(3), transposed=False)
    assert_array_almost_equal(x, [9, -5.4, -1.2])