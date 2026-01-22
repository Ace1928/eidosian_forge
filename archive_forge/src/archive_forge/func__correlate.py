import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def _correlate(in1, in2, mode='full', method='auto', convolution=False):
    quick_out = _st_core._check_conv_inputs(in1, in2, mode, convolution)
    if quick_out is not None:
        return quick_out
    if method not in ('auto', 'direct', 'fft'):
        raise ValueError('acceptable methods are "auto", "direct", or "fft"')
    if method == 'auto':
        method = choose_conv_method(in1, in2, mode=mode)
    if method == 'direct':
        return _st_core._direct_correlate(in1, in2, mode, in1.dtype, convolution)
    if not convolution:
        in2 = _st_core._reverse(in2).conj()
    inputs_swapped = _st_core._inputs_swap_needed(mode, in1.shape, in2.shape)
    if inputs_swapped:
        in1, in2 = (in2, in1)
    out = fftconvolve(in1, in2, mode)
    result_type = cupy.result_type(in1, in2)
    if result_type.kind in 'ui':
        out = out.round()
    out = out.astype(result_type, copy=False)
    return out