import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_vectorize_parallel_true_nopython_true_no_warnings(self):
    with _catch_numba_deprecation_warnings() as w:

        @vectorize('float64(float64)', target='parallel', nopython=True)
        def foo(x):
            return x + 1
    self.assertFalse(w)