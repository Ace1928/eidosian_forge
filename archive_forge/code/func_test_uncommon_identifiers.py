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
def test_uncommon_identifiers(self):

    @njit
    def foo():
        d = {'0': np.ones(5), '1': 4}
        return len(d)
    self.assertPreciseEqual(foo(), foo.py_func())

    @njit
    def bar():
        d = {'+': np.ones(5), 'x--': 4}
        return len(d)
    self.assertPreciseEqual(bar(), bar.py_func())