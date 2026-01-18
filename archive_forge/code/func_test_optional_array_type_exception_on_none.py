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
def test_optional_array_type_exception_on_none(self):
    self.disable_leak_check()

    @njit
    def arr_expr(x, y):
        return x + y

    @njit
    def do_call(x, y):
        if y[0] > 0:
            z = None
        else:
            z = y
        return arr_expr(x, z)
    args = (np.arange(5), np.arange(1.0, 5.0))
    with self.assertRaises(TypeError) as raises:
        do_call(*args)
    excstr = str(raises.exception)
    self.assertIn('expected array(float64,', excstr)
    self.assertIn('got None', excstr)
    s = arr_expr.signatures
    oty = s[0][1]
    self.assertTrue(isinstance(oty, types.Optional))
    self.assertTrue(isinstance(oty.type, types.Array))
    self.assertTrue(isinstance(oty.type.dtype, types.Float))