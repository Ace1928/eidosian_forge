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
def test_simple_cols(self):
    a = array([[1, 2, 3], [4, 5, 6]], dtype=float)
    a_pinv = pinv(a)
    expected = array([[-0.94444444, 0.44444444], [-0.11111111, 0.11111111], [0.72222222, -0.22222222]])
    assert_array_almost_equal(a_pinv, expected)