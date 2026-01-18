import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_not(self):
    pyfunc = return_not
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array('ab'), ())
    self._test(pyfunc, cfunc, np.array(b'ab'), ())
    self._test(pyfunc, cfunc, (b'ab',), 0)
    self._test(pyfunc, cfunc, np.array(''), ())
    self._test(pyfunc, cfunc, np.array(b''), ())
    self._test(pyfunc, cfunc, (b'',), 0)