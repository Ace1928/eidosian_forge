from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_weaktype(self):
    d = Dummy()
    e = Dummy()
    a = types.Dispatcher(d)
    b = types.Dispatcher(d)
    c = types.Dispatcher(e)
    self.assertIs(a.dispatcher, d)
    self.assertIs(b.dispatcher, d)
    self.assertIs(c.dispatcher, e)
    self.assertTrue(a == b)
    self.assertFalse(a != b)
    self.assertTrue(a != c)
    self.assertFalse(a == c)
    z = types.int8
    self.assertFalse(a == z)
    self.assertFalse(b == z)
    self.assertFalse(c == z)
    self.assertTrue(a != z)
    self.assertTrue(b != z)
    self.assertTrue(c != z)
    s = set([a, b, c])
    self.assertEqual(len(s), 2)
    self.assertIn(a, s)
    self.assertIn(b, s)
    self.assertIn(c, s)
    d = e = None
    gc.collect()
    with self.assertRaises(ReferenceError):
        a.dispatcher
    with self.assertRaises(ReferenceError):
        b.dispatcher
    with self.assertRaises(ReferenceError):
        c.dispatcher
    self.assertFalse(a == b)
    self.assertFalse(a == c)
    self.assertFalse(b == c)
    self.assertFalse(a == z)
    self.assertTrue(a != b)
    self.assertTrue(a != c)
    self.assertTrue(b != c)
    self.assertTrue(a != z)