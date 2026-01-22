import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
@jit.rawkernel()
def _check_prominence_invalid(n, peaks, left_bases, right_bases, out):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    i_min = left_bases[tid]
    i_max = right_bases[tid]
    peak = peaks[tid]
    valid = 0 <= i_min and i_min <= peak and (peak <= i_max) and (i_max < n)
    out[tid] = not valid