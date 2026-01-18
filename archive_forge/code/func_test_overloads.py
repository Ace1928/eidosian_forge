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
def test_overloads(self):
    JitList = jitclass({'x': types.List(types.intp)})(self.PyList)
    py_funcs = [lambda x: abs(x), lambda x: x.__abs__(), lambda x: bool(x), lambda x: x.__bool__(), lambda x: complex(x), lambda x: x.__complex__(), lambda x: 0 in x, lambda x: x.__contains__(0), lambda x: float(x), lambda x: x.__float__(), lambda x: int(x), lambda x: x.__int__(), lambda x: len(x), lambda x: x.__len__(), lambda x: str(x), lambda x: x.__str__(), lambda x: 1 if x else 0]
    jit_funcs = [njit(f) for f in py_funcs]
    py_list = self.PyList()
    jit_list = JitList()
    for py_f, jit_f in zip(py_funcs, jit_funcs):
        self.assertSame(py_f(py_list), py_f(jit_list))
        self.assertSame(py_f(py_list), jit_f(jit_list))
    py_list.append(2)
    jit_list.append(2)
    for py_f, jit_f in zip(py_funcs, jit_funcs):
        self.assertSame(py_f(py_list), py_f(jit_list))
        self.assertSame(py_f(py_list), jit_f(jit_list))
    py_list.append(-5)
    jit_list.append(-5)
    for py_f, jit_f in zip(py_funcs, jit_funcs):
        self.assertSame(py_f(py_list), py_f(jit_list))
        self.assertSame(py_f(py_list), jit_f(jit_list))
    py_list.clear()
    jit_list.clear()
    for py_f, jit_f in zip(py_funcs, jit_funcs):
        self.assertSame(py_f(py_list), py_f(jit_list))
        self.assertSame(py_f(py_list), jit_f(jit_list))