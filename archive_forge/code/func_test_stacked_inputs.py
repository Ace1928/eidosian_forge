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
@pytest.mark.parametrize('size', [(3, 4), (4, 3), (4, 4), (3, 0), (0, 3)])
@pytest.mark.parametrize('outer_size', [(2, 2), (2,), (2, 3, 4)])
@pytest.mark.parametrize('dt', [np.single, np.double, np.csingle, np.cdouble])
def test_stacked_inputs(self, outer_size, size, dt):
    A = np.random.normal(size=outer_size + size).astype(dt)
    B = np.random.normal(size=outer_size + size).astype(dt)
    self.check_qr_stacked(A)
    self.check_qr_stacked(A + 1j * B)