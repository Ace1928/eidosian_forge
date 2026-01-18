from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def test_tanhf_compile_ptx(self):

    def tanh_kernel(r, x):
        r[0] = tanh(x)

    def tanh_common_test(cc, criterion):
        fastptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32), fastmath=True, cc=cc)
        precptx, _ = compile_ptx(tanh_kernel, (float32[::1], float32), cc=cc)
        criterion.check(self, fastptx, precptx)
    tanh_common_test(cc=(7, 5), criterion=FastMathCriterion(fast_expected=['tanh.approx.f32 '], prec_unexpected=['tanh.approx.f32 ']))
    tanh_common_test(cc=(7, 0), criterion=FastMathCriterion(fast_expected=['ex2.approx.ftz.f32 ', 'rcp.approx.ftz.f32 '], prec_unexpected=['tanh.approx.f32 ']))