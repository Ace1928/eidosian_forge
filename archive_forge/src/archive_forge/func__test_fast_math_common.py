from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
def _test_fast_math_common(self, pyfunc, sig, device, criterion):
    fastver = cuda.jit(sig, device=device, fastmath=True)(pyfunc)
    precver = cuda.jit(sig, device=device)(pyfunc)
    criterion.check(self, fastver.inspect_asm(sig), precver.inspect_asm(sig))
    fastptx, _ = compile_ptx_for_current_device(pyfunc, sig, device=device, fastmath=True)
    precptx, _ = compile_ptx_for_current_device(pyfunc, sig, device=device)
    criterion.check(self, fastptx, precptx)