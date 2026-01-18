import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def test_unpack_tuple_of_arrays(self):
    tup = tuple((np.zeros(i + 1) for i in range(2)))
    tupty = typeof(tup)
    pyfunc = unpack_arbitrary
    cfunc = njit((tupty,))(pyfunc)
    self.assertPreciseEqual(cfunc(tup), pyfunc(tup))