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
def test_vector_return_type(self):
    a = np.array([1, 0, 1])
    exact_types = np.typecodes['AllInteger']
    inexact_types = np.typecodes['AllFloat']
    all_types = exact_types + inexact_types
    for each_type in all_types:
        at = a.astype(each_type)
        an = norm(at, -np.inf)
        self.check_dtype(at, an)
        assert_almost_equal(an, 0.0)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'divide by zero encountered')
            an = norm(at, -1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 0.0)
        an = norm(at, 0)
        self.check_dtype(at, an)
        assert_almost_equal(an, 2)
        an = norm(at, 1)
        self.check_dtype(at, an)
        assert_almost_equal(an, 2.0)
        an = norm(at, 2)
        self.check_dtype(at, an)
        assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 2.0))
        an = norm(at, 4)
        self.check_dtype(at, an)
        assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 4.0))
        an = norm(at, np.inf)
        self.check_dtype(at, an)
        assert_almost_equal(an, 1.0)