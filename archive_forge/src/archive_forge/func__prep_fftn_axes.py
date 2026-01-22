import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _prep_fftn_axes(ndim, s=None, axes=None, value_type='C2C'):
    """Configure axes argument for an n-dimensional FFT.

    The axes to be transformed are returned in ascending order.
    """
    if s is not None and axes is not None and (len(s) != len(axes)):
        raise ValueError('Shape and axes have different lengths.')
    if axes is None:
        if s is None:
            dim = ndim
        else:
            dim = len(s)
        axes = tuple([i + ndim for i in range(-dim, 0)])
        axes_sorted = axes
    else:
        axes = tuple(axes)
        if not axes:
            return ((), ())
        if _reduce(min, axes) < -ndim or _reduce(max, axes) > ndim - 1:
            raise ValueError('The specified axes exceed the array dimensions.')
        if value_type == 'C2C':
            axes_sorted = tuple(sorted([ax % ndim for ax in axes]))
        else:
            axes_sorted = sorted([ax % ndim for ax in axes[:-1]])
            axes_sorted.append(axes[-1] % ndim)
            axes_sorted = tuple(axes_sorted)
    return (axes, axes_sorted)