import numpy as np
import unittest
from numba import jit, njit
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.tests.support import no_pyobj_flags as nullary_no_pyobj_flags
from numba.tests.support import force_pyobj_flags as nullary_force_pyobj_flags
def unpack_range_too_large():
    a, b, c = range(4)
    return a + b + c