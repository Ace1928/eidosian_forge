import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def _fft_convolve(a1, a2, mode):
    offset = 0
    if a1.shape[-1] < a2.shape[-1]:
        a1, a2 = (a2, a1)
        offset = 1 - a2.shape[-1] % 2
    if a1.dtype.kind == 'c' or a2.dtype.kind == 'c':
        fft, ifft = (cupy.fft.fft, cupy.fft.ifft)
    else:
        fft, ifft = (cupy.fft.rfft, cupy.fft.irfft)
    dtype = cupy.result_type(a1, a2)
    n1, n2 = (a1.shape[-1], a2.shape[-1])
    out_size = cupyx.scipy.fft.next_fast_len(n1 + n2 - 1)
    fa1 = fft(a1, out_size)
    fa2 = fft(a2, out_size)
    out = ifft(fa1 * fa2, out_size)
    if mode == 'full':
        start, end = (0, n1 + n2 - 1)
    elif mode == 'same':
        start = (n2 - 1) // 2 + offset
        end = start + n1
    elif mode == 'valid':
        start, end = (n2 - 1, n1)
    else:
        raise ValueError('acceptable mode flags are `valid`, `same`, or `full`.')
    out = out[..., start:end]
    if dtype.kind in 'iu':
        out = cupy.around(out)
    return out.astype(dtype, copy=False)