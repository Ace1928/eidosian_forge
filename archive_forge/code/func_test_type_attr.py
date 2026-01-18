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
def test_type_attr(self):

    def conv(arr, val):
        return arr.dtype.type(val)
    jit_conv = jit(nopython=True)(conv)

    def assert_matches(arr, val, exact):
        expect = conv(arr, val)
        got = jit_conv(arr, val)
        self.assertPreciseEqual(expect, exact)
        self.assertPreciseEqual(typeof(expect), typeof(got))
        self.assertPreciseEqual(expect, got)
    arr = np.zeros(5)
    assert_matches(arr.astype(np.intp), 1.2, 1)
    assert_matches(arr.astype(np.float64), 1.2, 1.2)
    assert_matches(arr.astype(np.complex128), 1.2, 1.2 + 0j)
    assert_matches(arr.astype(np.complex128), 1.2j, 1.2j)