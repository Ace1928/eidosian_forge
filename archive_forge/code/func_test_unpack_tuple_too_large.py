import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def test_unpack_tuple_too_large(self):
    self.check_unpack_error(unpack_tuple_too_large)
    self.check_unpack_error(unpack_heterogeneous_tuple_too_large)