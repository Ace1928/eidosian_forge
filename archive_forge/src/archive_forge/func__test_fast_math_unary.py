from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def _test_fast_math_unary(self, op, criterion: FastMathCriterion):

    def kernel(r, x):
        r[0] = op(x)

    def device_function(x):
        return op(x)
    self._test_fast_math_common(kernel, (float32[::1], float32), device=False, criterion=criterion)
    self._test_fast_math_common(device_function, (float32,), device=True, criterion=criterion)