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
def test_str_item_refcount_replace(self):

    @njit
    def foo():
        i, j = ('ab', 'c')
        a = i + j
        m, n = ('zy', 'x')
        z = m + n
        l = List.empty_list(types.unicode_type)
        l.append(a)
        l[0] = z
        ra, rz = (get_refcount(a), get_refcount(z))
        return (l, ra, rz)
    l, ra, rz = foo()
    self.assertEqual(l[0], 'zyx')
    self.assertEqual(ra, 1)
    self.assertEqual(rz, 2)