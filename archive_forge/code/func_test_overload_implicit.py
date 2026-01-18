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
def test_overload_implicit(self):

    def do_this(x, y):
        return x + y

    @njit
    def hidden(x, y):
        return literally(x) + y

    @overload(do_this)
    def ov_do_this(x, y):
        if isinstance(x, types.Integer):
            return lambda x, y: hidden(x, y)

    @njit
    def foo(a, b):
        return do_this(a, b)
    a = 123
    b = 321
    r = foo(a, b)
    self.assertEqual(r, a + b)
    [type_a, type_b] = foo.signatures[0]
    self.assertIsInstance(type_a, types.Literal)
    self.assertEqual(type_a.literal_value, a)
    self.assertNotIsInstance(type_b, types.Literal)