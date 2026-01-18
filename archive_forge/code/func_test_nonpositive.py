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
def test_nonpositive(self):
    a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    a = np.dot(a, a.T)
    u, s, vt = np.linalg.svd(a)
    s[0] *= -1
    a = np.dot(u * s, vt)
    a_pinv = pinv(a)
    a_pinvh = pinvh(a)
    assert_array_almost_equal(a_pinv, a_pinvh)