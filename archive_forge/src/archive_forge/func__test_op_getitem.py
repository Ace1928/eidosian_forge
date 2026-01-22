import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def _test_op_getitem(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array([1, 2]), 0, 1)
    self._test(pyfunc, cfunc, '12', 0, 1)
    self._test(pyfunc, cfunc, b'12', 0, 1)
    self._test(pyfunc, cfunc, np.array(b'12'), (), ())
    self._test(pyfunc, cfunc, np.array('1234'), (), ())
    self._test(pyfunc, cfunc, np.array([b'1', b'2']), 0, 0)
    self._test(pyfunc, cfunc, np.array([b'1', b'2']), 0, 1)
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0, 0)
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1, 1)
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 0, 1)
    self._test(pyfunc, cfunc, np.array([b'12', b'3']), 1, 0)
    self._test(pyfunc, cfunc, np.array(['1', '2']), 0, 0)
    self._test(pyfunc, cfunc, np.array(['1', '2']), 0, 1)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 0, 0)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 1, 1)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 0, 1)
    self._test(pyfunc, cfunc, np.array(['12', '3']), 1, 0)