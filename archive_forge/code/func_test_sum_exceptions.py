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
def test_sum_exceptions(self):
    sum_default = njit(sum_usecase)
    sum_kwarg = njit(sum_kwarg_usecase)
    msg = "sum() can't sum {}"
    with self.assertRaises(errors.TypingError) as raises:
        sum_kwarg((1, 2, 3), 'a')
    self.assertIn(msg.format('strings'), str(raises.exception))
    with self.assertRaises(errors.TypingError) as raises:
        sum_kwarg((1, 2, 3), b'123')
    self.assertIn(msg.format('bytes'), str(raises.exception))
    with self.assertRaises(errors.TypingError) as raises:
        sum_kwarg((1, 2, 3), bytearray(b'123'))
    self.assertIn(msg.format('bytearray'), str(raises.exception))
    with self.assertRaises(errors.TypingError) as raises:
        sum_default('abcd')
    self.assertIn('No implementation', str(raises.exception))