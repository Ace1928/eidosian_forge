import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_ljust2(self):
    pyfunc = return_ljust2
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('1 2 3 4'), (), 40, np.array('='), ())
    self._test(pyfunc, cfunc, np.array('1 2 3 4'), (), 40, ('=',), 0)
    self._test(pyfunc, cfunc, ('1 2 3 4',), 0, 40, np.array('='), ())
    self._test(pyfunc, cfunc, np.array(b'1 2 3 4'), (), 40, np.array(b'='), ())
    self._test(pyfunc, cfunc, np.array(b'1 2 3 4'), (), 40, (b'=',), 0)
    self._test(pyfunc, cfunc, (b'1 2 3 4',), 0, 40, np.array(b'='), ())