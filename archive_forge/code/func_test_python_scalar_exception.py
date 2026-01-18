import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def test_python_scalar_exception(self):
    intty = getattr(np, 'int{}'.format(types.intp.bitwidth))

    @njit
    def myview():
        a = 1
        a.view(intty)
    with self.assertRaises(TypingError) as e:
        myview()
    self.assertIn("'view' can only be called on NumPy dtypes, try wrapping the variable 'a' with 'np.<dtype>()'", str(e.exception))