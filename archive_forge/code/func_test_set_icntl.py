import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_set_icntl(self):
    ma57 = MA57Interface()
    ma57.set_icntl(5, 4)
    ma57.set_icntl(8, 1)
    icntl5 = ma57.get_icntl(5)
    icntl8 = ma57.get_icntl(8)
    self.assertEqual(icntl5, 4)
    self.assertEqual(icntl8, 1)
    with self.assertRaisesRegex(TypeError, 'must be an integer'):
        ma57.set_icntl(1.0, 0)
    with self.assertRaisesRegex(IndexError, 'is out of range'):
        ma57.set_icntl(100, 0)
    with self.assertRaises(ctypes.ArgumentError):
        ma57.set_icntl(1, 0.0)