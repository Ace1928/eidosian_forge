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
def test_power_is_one(self, dt):

    def tz(mat):
        mz = matrix_power(mat, 1)
        assert_equal(mz, mat)
        assert_equal(mz.dtype, mat.dtype)
    for mat in self.rshft_all:
        tz(mat.astype(dt))
        if dt != object:
            tz(self.stacked.astype(dt))