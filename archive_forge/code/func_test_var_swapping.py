import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_var_swapping(self, flags=force_pyobj_jit_opt):
    pyfunc = var_swapping
    cfunc = jit((types.int32,) * 5, **flags)(pyfunc)
    args = tuple(range(0, 10, 2))
    self.assertPreciseEqual(pyfunc(*args), cfunc(*args))