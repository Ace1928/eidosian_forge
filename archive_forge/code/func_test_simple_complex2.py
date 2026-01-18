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
def test_simple_complex2(self):
    a = np.array([[-1.34 + 2.55j, 0.28 + 3.17j, -6.39 - 2.2j, 0.72 - 0.92j], [-1.7 - 14.1j, 33.1 - 1.5j, -1.5 + 13.4j, 12.9 + 13.8j], [-3.29 - 2.39j, -1.91 + 4.42j, -0.14 - 1.35j, 1.72 + 1.35j], [2.41 + 0.39j, -0.56 + 1.47j, -0.83 - 0.69j, -1.96 + 0.67j]])
    b = np.array([[26.26 + 51.78j, 31.32 - 6.7j], [64.3 - 86.8j, 158.6 - 14.2j], [-5.75 + 25.31j, -2.15 + 30.19j], [1.16 + 2.57j, -2.56 + 7.55j]])
    x = solve(a, b)
    assert_array_almost_equal(x, np.array([[1 + 1j, -1 - 2j], [2 - 3j, 5 + 1j], [-4 - 5j, -3 + 4j], [6j, 2 - 3j]]))