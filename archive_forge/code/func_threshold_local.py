import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def threshold_local(image, block_size=3, method='gaussian', offset=0, mode='reflect', param=None, cval=0):
    """Compute a threshold mask image based on local pixel neighborhood.

    Also known as adaptive or dynamic thresholding. The threshold value is
    the weighted mean for the local neighborhood of a pixel subtracted by a
    constant. Alternatively the threshold can be determined dynamically by a
    given function, using the 'generic' method.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Grayscale input image.
    block_size : int or sequence of int
        Odd size of pixel neighborhood which is used to calculate the
        threshold value (e.g. 3, 5, 7, ..., 21, ...).
    method : {'generic', 'gaussian', 'mean', 'median'}, optional
        Method used to determine adaptive threshold for local neighborhood in
        weighted mean image.

        * 'generic': use custom function (see ``param`` parameter)
        * 'gaussian': apply gaussian filter (see ``param`` parameter for custom                      sigma value)
        * 'mean': apply arithmetic mean filter
        * 'median': apply median rank filter

        By default, the 'gaussian' method is used.
    offset : float, optional
        Constant subtracted from weighted mean of neighborhood to calculate
        the local threshold value. Default offset is 0.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
        Default is 'reflect'.
    param : {int, function}, optional
        Either specify sigma for 'gaussian' method or function object for
        'generic' method. This functions takes the flat array of local
        neighborhood as a single argument and returns the calculated
        threshold for the centre pixel.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.

    Returns
    -------
    threshold : (M, N[, ...]) ndarray
        Threshold image. All pixels in the input image higher than the
        corresponding pixel in the threshold image are considered foreground.

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()[:50, :50]
    >>> binary_image1 = image > threshold_local(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = image > threshold_local(image, 15, 'generic',
    ...                                         param=func)

    """
    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError('len(block_size) must equal image.ndim.')
    block_size = tuple(block_size)
    if any((b % 2 == 0 for b in block_size)):
        raise ValueError(f'block_size must be odd! Given block_size {block_size} contains even values.')
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    thresh_image = np.zeros(image.shape, dtype=float_dtype)
    if method == 'generic':
        ndi.generic_filter(image, param, block_size, output=thresh_image, mode=mode, cval=cval)
    elif method == 'gaussian':
        if param is None:
            sigma = tuple([(b - 1) / 6.0 for b in block_size])
        else:
            sigma = param
        gaussian(image, sigma=sigma, out=thresh_image, mode=mode, cval=cval)
    elif method == 'mean':
        ndi.uniform_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode, cval=cval)
    else:
        raise ValueError('Invalid method specified. Please use `generic`, `gaussian`, `mean`, or `median`.')
    return thresh_image - offset