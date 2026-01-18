import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_return_len(self):
    pyfunc = return_len
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array(''), ())
    self._test(pyfunc, cfunc, np.array(b''), ())
    self._test(pyfunc, cfunc, np.array(b'12'), ())
    self._test(pyfunc, cfunc, np.array('12'), ())
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 0)
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 1)