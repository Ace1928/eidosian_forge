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
def test_stacked_singular(self):
    np.random.seed(1234)
    A = np.random.rand(2, 2, 2, 2)
    A[0, 0] = 0
    A[1, 1] = 0
    for p in (None, 1, 2, 'fro', -1, -2):
        c = linalg.cond(A, p)
        assert_equal(c[0, 0], np.inf)
        assert_equal(c[1, 1], np.inf)
        assert_(np.isfinite(c[0, 1]))
        assert_(np.isfinite(c[1, 0]))