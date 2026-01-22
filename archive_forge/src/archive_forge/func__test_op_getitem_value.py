import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def _test_op_getitem_value(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    self._test(pyfunc, cfunc, np.array([1, 2]), 0, 1)
    self._test(pyfunc, cfunc, '12', 0, '1')
    self._test(pyfunc, cfunc, '12', 1, '3')
    self._test(pyfunc, cfunc, np.array('1234'), (), '1234')
    self._test(pyfunc, cfunc, np.array(['1234']), 0, '1234')
    self._test(pyfunc, cfunc, np.array(['1234']), 0, 'abc')
    self._test(pyfunc, cfunc, np.array(b'12'), (), b'12')
    self._test(pyfunc, cfunc, np.array([b'12']), 0, b'12')
    self._test(pyfunc, cfunc, np.array([b'12']), 0, b'a')