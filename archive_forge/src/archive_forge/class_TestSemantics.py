import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
class TestSemantics(MemoryLeakMixin, unittest.TestCase):

    def test_division_by_zero(self):
        pyfunc = div_add
        cfunc = njit(pyfunc)
        a = np.float64([0.0, 1.0, float('inf')])
        b = np.float64([0.0, 0.0, 1.0])
        c = np.ones_like(a)
        expect = pyfunc(a, b, c)
        got = cfunc(a, b, c)
        np.testing.assert_array_equal(expect, got)