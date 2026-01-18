import warnings
import unittest
from contextlib import contextmanager
from numba import jit, vectorize, guvectorize
from numba.core.errors import (NumbaDeprecationWarning,
from numba.tests.support import TestCase, needs_setuptools
@TestCase.run_test_in_subprocess
def test_vectorize_calling_jit_with_nopython_false_warns_from_jit(self):
    with _catch_numba_deprecation_warnings() as w:

        @vectorize('float64(float64)', forceobj=True)
        def foo(x):
            return bar(x + 1)

        def bar(*args):
            pass
    self.assertFalse(w)