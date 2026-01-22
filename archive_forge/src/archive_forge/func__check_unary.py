import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def _check_unary(self, jitfunc, *args):
    pyfunc = jitfunc.py_func
    self.assertPreciseEqual(jitfunc(*args), pyfunc(*args))