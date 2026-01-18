import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_vectorize_missing_nopython_kwarg_not_reported(self):
    with _catch_numba_deprecation_warnings() as w:

        @vectorize('float64(float64)')
        def foo(a):
            return a + 1
    self.assertFalse(w)