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
@skip_parfors_unsupported
def test_unsigned_prange(self):

    @njit(parallel=True)
    def foo(a):
        r = types.uint64(3)
        s = types.uint64(0)
        for i in prange(r):
            s = s + a[i]
        return s
    a = List.empty_list(types.uint64)
    a.append(types.uint64(12))
    a.append(types.uint64(1))
    a.append(types.uint64(7))
    self.assertEqual(foo(a), 20)