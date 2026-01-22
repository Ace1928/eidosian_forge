import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class TestTensorinv:

    @pytest.mark.parametrize('arr, ind', [(np.ones((4, 6, 8, 2)), 2), (np.ones((3, 3, 2)), 1)])
    def test_non_square_handling(self, arr, ind):
        with assert_raises(LinAlgError):
            linalg.tensorinv(arr, ind=ind)

    @pytest.mark.parametrize('shape, ind', [((4, 6, 8, 3), 2), ((24, 8, 3), 1)])
    def test_tensorinv_shape(self, shape, ind):
        a = np.eye(24)
        a.shape = shape
        ainv = linalg.tensorinv(a=a, ind=ind)
        expected = a.shape[ind:] + a.shape[:ind]
        actual = ainv.shape
        assert_equal(actual, expected)

    @pytest.mark.parametrize('ind', [0, -2])
    def test_tensorinv_ind_limit(self, ind):
        a = np.eye(24)
        a.shape = (4, 6, 8, 3)
        with assert_raises(ValueError):
            linalg.tensorinv(a=a, ind=ind)

    def test_tensorinv_result(self):
        a = np.eye(24)
        a.shape = (24, 8, 3)
        ainv = linalg.tensorinv(a, ind=1)
        b = np.ones(24)
        assert_allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))