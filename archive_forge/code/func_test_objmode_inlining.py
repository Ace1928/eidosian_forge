import unittest
from numba.tests.support import TestCase
import sys
import operator
import numpy as np
import numpy
from numba import jit, njit, typed
from numba.core import types, utils
from numba.core.errors import TypingError, LoweringError
from numba.core.types.functions import _header_lead
from numba.np.numpy_support import numpy_version
from numba.tests.support import tag, _32bit, captured_stdout
def test_objmode_inlining(self):

    def objmode_func(y):
        z = object()
        inlined = [x for x in y]
        return inlined
    cfunc = jit(forceobj=True)(objmode_func)
    t = [1, 2, 3]
    expected = objmode_func(t)
    got = cfunc(t)
    self.assertPreciseEqual(expected, got)