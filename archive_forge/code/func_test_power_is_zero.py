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
def test_power_is_zero(self, dt):

    def tz(M):
        mz = matrix_power(M, 0)
        assert_equal(mz, identity_like_generalized(M))
        assert_equal(mz.dtype, M.dtype)
    for mat in self.rshft_all:
        tz(mat.astype(dt))
        if dt != object:
            tz(self.stacked.astype(dt))