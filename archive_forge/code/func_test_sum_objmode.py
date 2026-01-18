import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_sum_objmode(self, flags=forceobj_flags):
    pyfunc = sum_usecase
    cfunc = jit((types.Dummy('list'),), **flags)(pyfunc)
    x = range(10)
    self.assertPreciseEqual(cfunc(x), pyfunc(x))
    x = [x + x / 10.0 for x in range(10)]
    self.assertPreciseEqual(cfunc(x), pyfunc(x))
    x = [complex(x, x) for x in range(10)]
    self.assertPreciseEqual(cfunc(x), pyfunc(x))