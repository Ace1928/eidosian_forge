import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def _test_atomic_minmax_nan_val(self, func):
    cuda_func = cuda.jit('void(float64[:], float64[:,:])')(func)
    res = np.random.randint(0, 128, size=1).astype(np.float64)
    gold = res.copy()
    vals = np.zeros((1, 1), np.float64) + np.nan
    cuda_func[1, 1](res, vals)
    np.testing.assert_equal(res, gold)