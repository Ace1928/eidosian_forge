import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_np_flat(self):
    self.check_np_flat(gen_flat, **nopython_flags)