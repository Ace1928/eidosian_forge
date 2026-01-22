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
def _cmp_dance(self, expected, pa, pb, na, nb):
    self.assertEqual(cmp.py_func(pa, pb), expected)
    py_got = cmp.py_func(na, nb)
    self.assertEqual(py_got, expected)
    jit_got = cmp(na, nb)
    self.assertEqual(jit_got, expected)