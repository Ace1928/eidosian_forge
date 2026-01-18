from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
@cache(level=20)
def zero_crossings(y: np.ndarray, *, threshold: float=1e-10, ref_magnitude: Optional[Union[float, Callable]]=None, pad: bool=True, zero_pos: bool=True, axis: int=-1) -> np.ndarray:
    """Find the zero-crossings of a signal ``y``: indices ``i`` such that
    ``sign(y[i]) != sign(y[j])``.

    If ``y`` is multi-dimensional, then zero-crossings are computed along
    the specified ``axis``.

    Parameters
    ----------
    y : np.ndarray
        The input array

    threshold : float >= 0
        If non-zero, values where ``-threshold <= y <= threshold`` are
        clipped to 0.

    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to ``ref_magnitude``.

        If callable, the threshold is scaled relative to
        ``ref_magnitude(np.abs(y))``.

    pad : boolean
        If ``True``, then ``y[0]`` is considered a valid zero-crossing.

    zero_pos : boolean
        If ``True`` then the value 0 is interpreted as having positive sign.

        If ``False``, then 0, -1, and +1 all have distinct signs.

    axis : int
        Axis along which to compute zero-crossings.

    Returns
    -------
    zero_crossings : np.ndarray [shape=y.shape, dtype=boolean]
        Indicator array of zero-crossings in ``y`` along the selected axis.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> # Generate a time-series
    >>> y = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
    >>> y
    array([  0.000e+00,   9.694e-01,   4.759e-01,  -7.357e-01,
            -8.372e-01,   3.247e-01,   9.966e-01,   1.646e-01,
            -9.158e-01,  -6.142e-01,   6.142e-01,   9.158e-01,
            -1.646e-01,  -9.966e-01,  -3.247e-01,   8.372e-01,
             7.357e-01,  -4.759e-01,  -9.694e-01,  -9.797e-16])
    >>> # Compute zero-crossings
    >>> z = librosa.zero_crossings(y)
    >>> z
    array([ True, False, False,  True, False,  True, False, False,
            True, False,  True, False,  True, False, False,  True,
           False,  True, False,  True], dtype=bool)

    >>> # Stack y against the zero-crossing indicator
    >>> librosa.util.stack([y, z], axis=-1)
    array([[  0.000e+00,   1.000e+00],
           [  9.694e-01,   0.000e+00],
           [  4.759e-01,   0.000e+00],
           [ -7.357e-01,   1.000e+00],
           [ -8.372e-01,   0.000e+00],
           [  3.247e-01,   1.000e+00],
           [  9.966e-01,   0.000e+00],
           [  1.646e-01,   0.000e+00],
           [ -9.158e-01,   1.000e+00],
           [ -6.142e-01,   0.000e+00],
           [  6.142e-01,   1.000e+00],
           [  9.158e-01,   0.000e+00],
           [ -1.646e-01,   1.000e+00],
           [ -9.966e-01,   0.000e+00],
           [ -3.247e-01,   0.000e+00],
           [  8.372e-01,   1.000e+00],
           [  7.357e-01,   0.000e+00],
           [ -4.759e-01,   1.000e+00],
           [ -9.694e-01,   0.000e+00],
           [ -9.797e-16,   1.000e+00]])

    >>> # Find the indices of zero-crossings
    >>> np.nonzero(z)
    (array([ 0,  3,  5,  8, 10, 12, 15, 17, 19]),)
    """
    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))
    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude
    yi = y.swapaxes(-1, axis)
    z = np.empty_like(y, dtype=bool)
    zi = z.swapaxes(-1, axis)
    _zc_wrapper(yi, threshold, zero_pos, zi)
    zi[..., 0] = pad
    return z