import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_popc_u4(self):
    compiled = cuda.jit('void(int32[:], uint32)')(simple_popc)
    ary = np.zeros(1, dtype=np.int32)
    compiled[1, 1](ary, 240)
    self.assertEqual(ary[0], 4)