import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Tests PTX emission')
def test_selp(self):
    sig = (int64[:], int64, int64[:])
    cu_branching_with_ifs = cuda.jit(sig)(branching_with_ifs)
    cu_branching_with_selps = cuda.jit(sig)(branching_with_selps)
    n = 32
    b = 6
    c = np.full(shape=32, fill_value=17, dtype=np.int64)
    expected = c.copy()
    expected[:5] = 3
    a = np.arange(n, dtype=np.int64)
    cu_branching_with_ifs[n, 1](a, b, c)
    ptx = cu_branching_with_ifs.inspect_asm(sig)
    self.assertEqual(2, len(re.findall('\\s+bra\\s+', ptx)))
    np.testing.assert_array_equal(a, expected, err_msg='branching')
    a = np.arange(n, dtype=np.int64)
    cu_branching_with_selps[n, 1](a, b, c)
    ptx = cu_branching_with_selps.inspect_asm(sig)
    self.assertEqual(0, len(re.findall('\\s+bra\\s+', ptx)))
    np.testing.assert_array_equal(a, expected, err_msg='selp')