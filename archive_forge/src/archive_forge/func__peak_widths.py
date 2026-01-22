import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases, check=False):
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    if prominences is None:
        raise TypeError('prominences must not be None')
    if left_bases is None:
        raise TypeError('left_bases must not be None')
    if right_bases is None:
        raise TypeError('right_bases must not be None')
    if not peaks.shape[0] == prominences.shape[0] == left_bases.shape[0] == right_bases.shape[0]:
        raise ValueError('arrays in `prominence_data` must have the same shape as `peaks`')
    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz
    if check and n > 0:
        invalid = cupy.zeros(n, dtype=cupy.bool_)
        _check_prominence_invalid((n_blocks,), (block_sz,), (x.shape[0], peaks, left_bases, right_bases, invalid))
        if cupy.any(invalid):
            raise ValueError('prominence data is invalid')
    widths = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    width_heights = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    left_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    right_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    peak_widths_kernel = _get_module_func(PEAKS_MODULE, 'peak_widths', x)
    peak_widths_kernel((n_blocks,), (block_sz,), (n, x, peaks, rel_height, prominences, left_bases, right_bases, widths, width_heights, left_ips, right_ips))
    return (widths, width_heights, left_ips, right_ips)