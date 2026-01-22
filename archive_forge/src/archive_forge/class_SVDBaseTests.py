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
class SVDBaseTests:
    hermitian = False

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        res = linalg.svd(x)
        U, S, Vh = (res.U, res.S, res.Vh)
        assert_equal(U.dtype, dtype)
        assert_equal(S.dtype, get_real_dtype(dtype))
        assert_equal(Vh.dtype, dtype)
        s = linalg.svd(x, compute_uv=False, hermitian=self.hermitian)
        assert_equal(s.dtype, get_real_dtype(dtype))