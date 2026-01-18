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
def test_tuple_not_in_mro(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        self.assertFalse(isinstance(x, types.BaseTuple))
        self.assertTrue(isinstance(x, types.LiteralStrKeyDict))
        return lambda x: ...

    @njit
    def foo():
        d = {'a': 1, 'b': 'c'}
        bar(d)
    foo()