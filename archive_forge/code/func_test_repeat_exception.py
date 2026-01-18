import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_repeat_exception(self):
    np_pyfunc = np_repeat
    np_nbfunc = njit(np_pyfunc)
    array_pyfunc = array_repeat
    array_nbfunc = njit(array_pyfunc)
    self.disable_leak_check()
    for pyfunc, nbfunc in ((np_pyfunc, np_nbfunc), (array_pyfunc, array_nbfunc)):
        with self.assertRaises(ValueError) as e:
            nbfunc(np.ones(1), -1)
        self.assertIn('negative dimensions are not allowed', str(e.exception))
        with self.assertRaises(TypingError) as e:
            nbfunc(np.ones(1), 1.0)
        self.assertIn('The repeats argument must be an integer or an array-like of integer dtype', str(e.exception))
        with self.assertRaises(ValueError) as e:
            nbfunc(np.ones(2), np.array([1, -1]))
        self.assertIn('negative dimensions are not allowed', str(e.exception))
        with self.assertRaises(ValueError) as e:
            nbfunc(np.ones(2), np.array([1, 1, 1]))
        self.assertIn('operands could not be broadcast together', str(e.exception))
        with self.assertRaises(ValueError) as e:
            nbfunc(np.ones(5), np.array([1, 1, 1, 1]))
        self.assertIn('operands could not be broadcast together', str(e.exception))
        with self.assertRaises(TypingError) as e:
            nbfunc(np.ones(2), [1.0, 1.0])
        self.assertIn('The repeats argument must be an integer or an array-like of integer dtype', str(e.exception))
        for rep in [True, 'a', '1']:
            with self.assertRaises(TypingError):
                nbfunc(np.ones(1), rep)