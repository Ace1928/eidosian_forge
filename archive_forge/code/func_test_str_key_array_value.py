import sys
import warnings
import numpy as np
from numba import njit, literally
from numba import int32, int64, float32, float64
from numba import typeof
from numba.typed import Dict, dictobject, List
from numba.typed.typedobjectutils import _sentry_safe_cast
from numba.core.errors import TypingError
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin, unittest,
from numba.experimental import jitclass
from numba.extending import overload
def test_str_key_array_value(self):
    np.random.seed(123)
    d = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    expect = []
    expect.append(np.random.random(10))
    d['mass'] = expect[-1]
    expect.append(np.random.random(20))
    d['velocity'] = expect[-1]
    for i in range(100):
        expect.append(np.random.random(i))
        d[str(i)] = expect[-1]
    self.assertEqual(len(d), len(expect))
    self.assertPreciseEqual(d['mass'], expect[0])
    self.assertPreciseEqual(d['velocity'], expect[1])
    for got, exp in zip(d.values(), expect):
        self.assertPreciseEqual(got, exp)
    self.assertTrue('mass' in d)
    self.assertTrue('velocity' in d)
    del d['mass']
    self.assertFalse('mass' in d)
    del d['velocity']
    self.assertFalse('velocity' in d)
    del expect[0:2]
    for i in range(90):
        k, v = d.popitem()
        w = expect.pop()
        self.assertPreciseEqual(v, w)
    expect.append(np.random.random(10))
    d['last'] = expect[-1]
    for got, exp in zip(d.values(), expect):
        self.assertPreciseEqual(got, exp)