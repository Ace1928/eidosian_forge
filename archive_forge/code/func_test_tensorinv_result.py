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
def test_tensorinv_result(self):
    a = np.eye(24)
    a.shape = (24, 8, 3)
    ainv = linalg.tensorinv(a, ind=1)
    b = np.ones(24)
    assert_allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))