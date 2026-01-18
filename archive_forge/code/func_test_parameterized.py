import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_parameterized(self):

    class MyClass(object):

        def __init__(self, value):
            self.value = value

    def create_my_class(value):
        cls = jitclass(MyClass, [('value', typeof(value))])
        return cls(value)
    a = create_my_class(123)
    self.assertEqual(a.value, 123)
    b = create_my_class(12.3)
    self.assertEqual(b.value, 12.3)
    c = create_my_class(np.array([123]))
    np.testing.assert_equal(c.value, [123])
    d = create_my_class(np.array([12.3]))
    np.testing.assert_equal(d.value, [12.3])