import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_except_reraise(self):

    @njit
    def udt():
        try:
            raise ValueError('ERROR')
        except Exception:
            raise
    with self.assertRaises(UnsupportedError) as raises:
        udt()
    self.assertIn('The re-raising of an exception is not yet supported.', str(raises.exception))