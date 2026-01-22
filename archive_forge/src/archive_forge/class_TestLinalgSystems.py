import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
class TestLinalgSystems(TestLinalgBase):
    """
    Base class for testing "system" solvers from np.linalg.
    Namely np.linalg.solve() and np.linalg.lstsq().
    """

    def assert_wrong_dimensions_1D(self, name, cfunc, args, la_prefix=True):
        prefix = 'np.linalg' if la_prefix else 'np'
        msg = '%s.%s() only supported on 1 and 2-D arrays' % (prefix, name)
        self.assert_error(cfunc, args, msg, errors.TypingError)

    def assert_dimensionally_invalid(self, cfunc, args):
        msg = 'Incompatible array sizes, system is not dimensionally valid.'
        self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)

    def assert_homogeneous_dtypes(self, name, cfunc, args):
        msg = 'np.linalg.%s() only supports inputs that have homogeneous dtypes.' % name
        self.assert_error(cfunc, args, msg, errors.TypingError)