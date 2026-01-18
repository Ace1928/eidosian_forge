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
def test_overload_explicit(self):

    def do_this(x, y):
        return x + y

    @overload(do_this)
    def ov_do_this(x, y):
        SentryLiteralArgs(['x']).for_function(ov_do_this).bind(x, y)
        return lambda x, y: x + y

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