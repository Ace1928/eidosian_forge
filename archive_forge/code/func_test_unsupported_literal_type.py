import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
def test_unsupported_literal_type(self):

    @njit
    def foo(a, b, c):
        return inner(a, c)

    @njit
    def inner(x, y):
        return x + literally(y)
    arr = np.arange(10)
    with self.assertRaises(errors.LiteralTypingError) as raises:
        foo(a=1, b=2, c=arr)
    self.assertIn('numpy.ndarray', str(raises.exception))