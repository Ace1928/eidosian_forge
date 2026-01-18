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
@pytest.mark.parametrize(['m', 'n'], [(3, 0), (0, 3), (0, 0)])
def test_qr_empty(self, m, n):
    k = min(m, n)
    a = np.empty((m, n))
    self.check_qr(a)
    h, tau = np.linalg.qr(a, mode='raw')
    assert_equal(h.dtype, np.double)
    assert_equal(tau.dtype, np.double)
    assert_equal(h.shape, (n, m))
    assert_equal(tau.shape, (k,))