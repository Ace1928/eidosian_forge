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
def test_spec_errors(self):
    spec1 = [('x', int), ('y', float32[:])]
    spec2 = [(1, int32), ('y', float32[:])]

    class Test(object):

        def __init__(self):
            pass
    with self.assertRaises(TypeError) as raises:
        jitclass(Test, spec1)
    self.assertIn('spec values should be Numba type instances', str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        jitclass(Test, spec2)
    self.assertEqual(str(raises.exception), 'spec keys should be strings, got 1')