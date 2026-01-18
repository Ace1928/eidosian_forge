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
def test_simple_sym_complexb(self):
    a = [[5, 2], [2, -4]]
    for b in ([1j, 0], [[1j, 1j], [0, 2]]):
        x = solve(a, b, assume_a='sym')
        assert_array_almost_equal(dot(a, x), b)