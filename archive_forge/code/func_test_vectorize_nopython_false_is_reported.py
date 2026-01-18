import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_vectorize_nopython_false_is_reported(self):
    with _catch_numba_deprecation_warnings() as w:

        @vectorize('float64(float64)', nopython=False)
        def foo(a):
            return a + 1
    msg = "The keyword argument 'nopython=False' was supplied"
    self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)