from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def _test_fast_math_binary(self, op, criterion: FastMathCriterion):

    def kernel(r, x, y):
        r[0] = op(x, y)

    def device(x, y):
        return op(x, y)
    self._test_fast_math_common(kernel, (float32[::1], float32, float32), device=False, criterion=criterion)
    self._test_fast_math_common(device, (float32, float32), device=True, criterion=criterion)