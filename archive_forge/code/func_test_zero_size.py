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
def test_zero_size(self):
    for a_shape, b_shape in (((0, 2), (0,)), ((0, 4), (0, 2)), ((4, 0), (4,)), ((4, 0), (4, 2))):
        b = np.ones(b_shape)
        x, residues, rank, s = lstsq(np.zeros(a_shape), b)
        assert_equal(x, np.zeros((a_shape[1],) + b_shape[1:]))
        residues_should_be = np.empty((0,)) if a_shape[1] else np.linalg.norm(b, axis=0) ** 2
        assert_equal(residues, residues_should_be)
        assert_(rank == 0, 'expected rank 0')
        assert_equal(s, np.empty((0,)))