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
class TestEigvals(EigvalsCases):

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, dtype)
        x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
        assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))

    def test_0_size(self):

        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        assert_(isinstance(res, np.ndarray))
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.eigvals(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal((0,), res.shape)
        assert_(isinstance(res, np.ndarray))