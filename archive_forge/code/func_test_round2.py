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
def test_round2(self, flags=forceobj_flags):
    pyfunc = round_usecase2
    for tp in (types.float64, types.float32):
        prec = 'single' if tp is types.float32 else 'exact'
        cfunc = jit((tp, types.int32), **flags)(pyfunc)
        for x in [0.0, 0.1, 0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 12.5, 15.0, 22.5]:
            for n in (-1, 0, 1, 2):
                self.assertPreciseEqual(cfunc(x, n), pyfunc(x, n), prec=prec)
                expected = pyfunc(-x, n)
                self.assertPreciseEqual(cfunc(-x, n), pyfunc(-x, n), prec=prec)