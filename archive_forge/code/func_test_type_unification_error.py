import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_type_unification_error(self):
    pyfunc = gen_unification_error
    with self.assertTypingError() as raises:
        jit((), **nopython_flags)(pyfunc)
    msg = "Can't unify yield type from the following types: complex128, none"
    self.assertIn(msg, str(raises.exception))