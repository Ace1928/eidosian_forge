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
def test_numeric_fallback(self):

    def py_c(x):
        return complex(x)

    def py_f(x):
        return float(x)

    def py_i(x):
        return int(x)
    jit_c = njit(py_c)
    jit_f = njit(py_f)
    jit_i = njit(py_i)

    @jitclass([])
    class FloatClass:

        def __init__(self):
            pass

        def __float__(self):
            return 3.1415
    obj = FloatClass()
    self.assertSame(py_c(obj), complex(3.1415))
    self.assertSame(jit_c(obj), complex(3.1415))
    self.assertSame(py_f(obj), 3.1415)
    self.assertSame(jit_f(obj), 3.1415)
    with self.assertRaises(TypeError) as e:
        py_i(obj)
    self.assertIn('int', str(e.exception))
    with self.assertRaises(TypingError) as e:
        jit_i(obj)
    self.assertIn('int', str(e.exception))

    @jitclass([])
    class IntClass:

        def __init__(self):
            pass

        def __int__(self):
            return 7
    obj = IntClass()
    self.assertSame(py_i(obj), 7)
    self.assertSame(jit_i(obj), 7)
    with self.assertRaises(TypeError) as e:
        py_c(obj)
    self.assertIn('complex', str(e.exception))
    with self.assertRaises(TypingError) as e:
        jit_c(obj)
    self.assertIn('complex', str(e.exception))
    with self.assertRaises(TypeError) as e:
        py_f(obj)
    self.assertIn('float', str(e.exception))
    with self.assertRaises(TypingError) as e:
        jit_f(obj)
    self.assertIn('float', str(e.exception))

    @jitclass([])
    class IndexClass:

        def __init__(self):
            pass

        def __index__(self):
            return 1
    obj = IndexClass()
    self.assertSame(py_c(obj), complex(1))
    self.assertSame(jit_c(obj), complex(1))
    self.assertSame(py_f(obj), 1.0)
    self.assertSame(jit_f(obj), 1.0)
    self.assertSame(py_i(obj), 1)
    self.assertSame(jit_i(obj), 1)

    @jitclass([])
    class FloatIntIndexClass:

        def __init__(self):
            pass

        def __float__(self):
            return 3.1415

        def __int__(self):
            return 7

        def __index__(self):
            return 1
    obj = FloatIntIndexClass()
    self.assertSame(py_c(obj), complex(3.1415))
    self.assertSame(jit_c(obj), complex(3.1415))
    self.assertSame(py_f(obj), 3.1415)
    self.assertSame(jit_f(obj), 3.1415)
    self.assertSame(py_i(obj), 7)
    self.assertSame(jit_i(obj), 7)