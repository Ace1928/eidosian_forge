import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
def _create_wrapped(self, pyfunc, intype, outtype):
    wrapped_func = cuda.jit(device=True)(pyfunc)

    @cuda.jit
    def cuda_wrapper_fn(arg, res):
        res[0] = wrapped_func(arg[0])

    def wrapper_fn(arg):
        argarray = np.zeros(1, dtype=intype)
        argarray[0] = arg
        resarray = np.zeros(1, dtype=outtype)
        cuda_wrapper_fn[1, 1](argarray, resarray)
        return resarray[0]
    return wrapper_fn