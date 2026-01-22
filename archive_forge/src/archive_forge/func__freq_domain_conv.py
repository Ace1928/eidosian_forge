import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    if not axes:
        return in1 * in2
    real = in1.dtype.kind != 'c' and in2.dtype.kind != 'c'
    fshape = [fft.next_fast_len(shape[a], real) for a in axes] if calc_fast_len else shape
    fftn, ifftn = (fft.rfftn, fft.irfftn) if real else (fft.fftn, fft.ifftn)
    sp1 = fftn(in1, fshape, axes=axes)
    sp2 = fftn(in2, fshape, axes=axes)
    out = ifftn(sp1 * sp2, fshape, axes=axes)
    return out[tuple((slice(x) for x in shape))] if calc_fast_len else out