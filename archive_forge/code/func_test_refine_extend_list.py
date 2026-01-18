import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_refine_extend_list(self):

    @njit
    def foo():
        a = List()
        b = List()
        for i in range(3):
            b.append(i)
        a.extend(b)
        return a
    expected = foo.py_func()
    got = foo()
    self.assertEqual(expected, got)
    self.assertEqual(list(got), [0, 1, 2])
    self.assertEqual(typeof(got).item_type, typeof(1))