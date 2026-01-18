import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_isascii(self):
    pyfunc = return_isascii
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('1234'), ())
    self._test(pyfunc, cfunc, np.array(['1234']), 0)
    self._test(pyfunc, cfunc, np.array('1234é'), ())
    self._test(pyfunc, cfunc, np.array(['1234é']), 0)