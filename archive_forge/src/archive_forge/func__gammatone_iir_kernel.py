import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import (
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
@jit.rawkernel()
def _gammatone_iir_kernel(fs, freq, b, a):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    EarQ = 9.26449
    minBW = 24.7
    erb = freq / EarQ + minBW
    T = 1.0 / fs
    bw = 2 * cupy.pi * 1.019 * erb
    fr = 2 * freq * cupy.pi * T
    bwT = bw * T
    g1 = -2 * cupy.exp(2j * fr) * T
    g2 = 2 * cupy.exp(-bwT + 1j * fr) * T
    g3 = cupy.sqrt(3 + 2 ** (3 / 2)) * cupy.sin(fr)
    g4 = cupy.sqrt(3 - 2 ** (3 / 2)) * cupy.sin(fr)
    g5 = cupy.exp(2j * fr)
    g = g1 + g2 * (cupy.cos(fr) - g4)
    g *= g1 + g2 * (cupy.cos(fr) + g4)
    g *= g1 + g2 * (cupy.cos(fr) - g3)
    g *= g1 + g2 * (cupy.cos(fr) + g3)
    g /= (-2 / cupy.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / cupy.exp(bwT)) ** 4
    g_act = cupy.abs(g)
    if tid == 0:
        b[tid] = T ** 4 / g_act
        a[tid] = 1
    elif tid == 1:
        b[tid] = -4 * T ** 4 * cupy.cos(fr) / cupy.exp(bw * T) / g_act
        a[tid] = -8 * cupy.cos(fr) / cupy.exp(bw * T)
    elif tid == 2:
        b[tid] = 6 * T ** 4 * cupy.cos(2 * fr) / cupy.exp(2 * bw * T) / g_act
        a[tid] = 4 * (4 + 3 * cupy.cos(2 * fr)) / cupy.exp(2 * bw * T)
    elif tid == 3:
        b[tid] = -4 * T ** 4 * cupy.cos(3 * fr) / cupy.exp(3 * bw * T) / g_act
        a[tid] = -8 * (6 * cupy.cos(fr) + cupy.cos(3 * fr))
        a[tid] /= cupy.exp(3 * bw * T)
    elif tid == 4:
        b[tid] = T ** 4 * cupy.cos(4 * fr) / cupy.exp(4 * bw * T) / g_act
        a[tid] = 2 * (18 + 16 * cupy.cos(2 * fr) + cupy.cos(4 * fr))
        a[tid] /= cupy.exp(4 * bw * T)
    elif tid == 5:
        a[tid] = -8 * (6 * cupy.cos(fr) + cupy.cos(3 * fr))
        a[tid] /= cupy.exp(5 * bw * T)
    elif tid == 6:
        a[tid] = 4 * (4 + 3 * cupy.cos(2 * fr)) / cupy.exp(6 * bw * T)
    elif tid == 7:
        a[tid] = -8 * cupy.cos(fr) / cupy.exp(7 * bw * T)
    elif tid == 8:
        a[tid] = cupy.exp(-8 * bw * T)