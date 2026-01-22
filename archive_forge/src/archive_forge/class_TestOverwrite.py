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
class TestOverwrite:

    def test_solve(self):
        assert_no_overwrite(solve, [(3, 3), (3,)])

    def test_solve_triangular(self):
        assert_no_overwrite(solve_triangular, [(3, 3), (3,)])

    def test_solve_banded(self):
        assert_no_overwrite(lambda ab, b: solve_banded((2, 1), ab, b), [(4, 6), (6,)])

    def test_solveh_banded(self):
        assert_no_overwrite(solveh_banded, [(2, 6), (6,)])

    def test_inv(self):
        assert_no_overwrite(inv, [(3, 3)])

    def test_det(self):
        assert_no_overwrite(det, [(3, 3)])

    def test_lstsq(self):
        assert_no_overwrite(lstsq, [(3, 2), (3,)])

    def test_pinv(self):
        assert_no_overwrite(pinv, [(3, 3)])

    def test_pinvh(self):
        assert_no_overwrite(pinvh, [(3, 3)])