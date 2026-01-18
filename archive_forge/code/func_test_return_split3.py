import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_split3(self):
    pyfunc = return_split3
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('1 2 3 4'), (), np.array(' '), (), 2)
    self._test(pyfunc, cfunc, np.array('1 2 3 4'), (), (' ',), 0, 2)
    self._test(pyfunc, cfunc, ('1 2 3 4',), 0, np.array(' '), (), 2)
    self._test(pyfunc, cfunc, np.array(b'1 2 3 4'), (), np.array(b' '), (), 2)
    self._test(pyfunc, cfunc, np.array(b'1 2 3 4'), (), (b' ',), 0, 2)
    self._test(pyfunc, cfunc, (b'1 2 3 4',), 0, np.array(b' '), (), 2)