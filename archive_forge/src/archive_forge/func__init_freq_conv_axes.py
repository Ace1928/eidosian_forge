import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    s1, s2 = (in1.shape, in2.shape)
    axes = _init_nd_and_axes(in1, axes)
    axes = [ax for ax in axes if s1[ax] != 1 and s2[ax] != 1]
    if sorted_axes:
        axes.sort()
    for ax, (dim1, dim2) in enumerate(zip(s1, s2)):
        if ax not in axes and dim1 != dim2 and (dim1 != 1) and (dim2 != 1):
            raise ValueError('incompatible shapes for in1 and in2: {} and {}'.format(s1, s2))
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        in1, in2 = (in2, in1)
    return (in1, in2, tuple(axes))