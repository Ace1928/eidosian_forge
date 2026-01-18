from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_stack_offset_error_when_has_no_return(self):
    from numba import jit
    import warnings

    def pyfunc(a):
        if a:
            for i in range(10):
                pass
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cfunc = jit(forceobj=True)(pyfunc)
        self.assertEqual(pyfunc(True), cfunc(True))