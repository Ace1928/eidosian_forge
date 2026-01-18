import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_ufunc_props(self):
    duadd = self.nopython_dufunc(pyuadd)
    self.assertEqual(duadd.nin, 2)
    self.assertEqual(duadd.nout, 1)
    self.assertEqual(duadd.nargs, duadd.nin + duadd.nout)
    self.assertEqual(duadd.ntypes, 0)
    self.assertEqual(duadd.types, [])
    self.assertEqual(duadd.identity, None)
    duadd(1, 2)
    self.assertEqual(duadd.ntypes, 1)
    self.assertEqual(duadd.ntypes, len(duadd.types))
    self.assertIsNone(duadd.signature)