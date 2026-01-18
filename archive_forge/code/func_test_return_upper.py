import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_upper(self):
    pyfunc = return_upper
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('abc'), ())
    self._test(pyfunc, cfunc, np.array(['abc']), 0)
    self._test(pyfunc, cfunc, np.array(b'abc'), ())
    self._test(pyfunc, cfunc, np.array([b'abc']), 0)