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
@jitclass({'x': types.float64})
class JitFloatUpdateWrapper(PyFloatWrapper):

    def __init__(self, value):
        self.x = value

    def __iadd__(self, other):
        return JitFloatUpdateWrapper(self.x + 2.718 * other.x)

    def __ifloordiv__(self, other):
        return JitFloatUpdateWrapper(self.x * 2.718 // other.x)

    def __imod__(self, other):
        return JitFloatUpdateWrapper(self.x % (other.x + 1))

    def __imul__(self, other):
        return JitFloatUpdateWrapper(self.x * other.x + 1)

    def __ipow__(self, other):
        return JitFloatUpdateWrapper(self.x ** other.x + 1)

    def __isub__(self, other):
        return JitFloatUpdateWrapper(self.x - 3.1415 * other.x)

    def __itruediv__(self, other):
        return JitFloatUpdateWrapper((self.x + 1) / other.x)