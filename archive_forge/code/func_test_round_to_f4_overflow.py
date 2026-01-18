import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Overflow behavior differs on CPython')
def test_round_to_f4_overflow(self):
    compiled = cuda.jit('void(float32[:], float32, int32)')(simple_round_to)
    ary = np.zeros(1, dtype=np.float32)
    val = np.finfo(np.float32).max
    ndigits = 300
    compiled[1, 1](ary, val, ndigits)
    self.assertEqual(ary[0], val)