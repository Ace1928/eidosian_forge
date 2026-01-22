import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _dct_or_dst_type3(x, n=None, axis=-1, norm=None, forward=True, dst=False, overwrite_x=False):
    """Forward DCT/DST-III (or inverse DCT/DST-II) along a single axis.

    Parameters
    ----------
    x : cupy.ndarray
        The data to transform.
    n : int
        The size of the transform. If None, ``x.shape[axis]`` is used.
    axis : int
        Axis along which the transform is applied.
    forward : bool
        Set true to indicate that this is a forward DCT-II as opposed to an
        inverse DCT-III (The difference between the two is only in the
        normalization factor).
    norm : {None, 'ortho', 'forward', 'backward'}
        The normalization convention to use.
    dst : bool
        If True, a discrete sine transform is computed rather than the discrete
        cosine transform.
    overwrite_x : bool
        Indicates that it is okay to overwrite x. In practice, the current
        implementation never performs the transform in-place.

    Returns
    -------
    y: cupy.ndarray
        The transformed array.

    """
    if axis < -x.ndim or axis >= x.ndim:
        raise numpy.AxisError('axis out of range')
    if axis < 0:
        axis += x.ndim
    if n is not None and n < 1:
        raise ValueError(f'invalid number of data points ({n}) specified')
    x = _cook_shape(x, (n,), (axis,), 'R2R')
    n = x.shape[axis]
    if norm == 'ortho':
        sl0_scale = 0.5 * math.sqrt(2)
        inorm = 'sqrt'
    elif norm == 'forward':
        sl0_scale = 0.5
        inorm = 'full' if forward else 'none'
    elif norm == 'backward' or norm is None:
        sl0_scale = 0.5
        inorm = 'none' if forward else 'full'
    else:
        raise ValueError(f'Invalid norm value "{norm}", should be "backward", "ortho" or "forward"')
    norm_factor = _get_dct_norm_factor(n, inorm=inorm, dct_type=3)
    dtype = cupy.promote_types(x, cupy.complex64)
    sl0 = [slice(None)] * x.ndim
    sl0[axis] = slice(1)
    if dst:
        if norm == 'ortho':
            float_dtype = cupy.promote_types(x.dtype, cupy.float32)
            if x.dtype != float_dtype:
                x = x.astype(float_dtype)
            elif not overwrite_x:
                x = x.copy()
            x[tuple(sl0)] *= math.sqrt(2)
            sl0_scale = 0.5
        slrev = [slice(None)] * x.ndim
        slrev[axis] = slice(None, None, -1)
        x = x[tuple(slrev)]
    tmp = _exp_factor_dct3(x, n, axis, dtype, norm_factor)
    x = x * tmp
    x[tuple(sl0)] *= sl0_scale
    x = _fft.ifft(x, n=n, axis=axis, overwrite_x=True)
    x = cupy.real(x)
    return _reshuffle_dct3(x, n, axis, dst)