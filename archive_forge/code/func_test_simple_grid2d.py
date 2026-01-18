import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_simple_grid2d(self):
    compiled = cuda.jit('void(int32[:,::1])')(simple_grid2d)
    ntid = (4, 3)
    nctaid = (5, 6)
    shape = (ntid[0] * nctaid[0], ntid[1] * nctaid[1])
    ary = np.empty(shape, dtype=np.int32)
    exp = ary.copy()
    compiled[nctaid, ntid](ary)
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            exp[i, j] = i + j
    self.assertTrue(np.all(ary == exp))