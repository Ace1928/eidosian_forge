import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def test_unpack_list(self):
    pyfunc = unpack_list
    cfunc = jit(forceobj=True)(pyfunc)
    l = [1, 2, 3]
    self.assertEqual(cfunc(l), pyfunc(l))