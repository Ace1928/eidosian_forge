import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
@skip_unless_scipy
def test_real_problem(self):

    @njit
    def foo():
        a = np.zeros((4, 4))
        try:
            chol = np.linalg.cholesky(a)
        except:
            print('CAUGHT')
            return chol
    with captured_stdout() as stdout:
        foo()
    self.assertEqual(stdout.getvalue().split(), ['CAUGHT'])