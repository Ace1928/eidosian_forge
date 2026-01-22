import itertools
import functools
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type
from ..metrics import mean_squared_error
from ..util import img_as_float
def _interpolate_image(image, *, multichannel=False):
    """Replacing each pixel in ``image`` with the average of its neighbors.

    Parameters
    ----------
    image : ndarray
        Input data to be interpolated.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    interp : ndarray
        Interpolated version of `image`.
    """
    spatialdims = image.ndim if not multichannel else image.ndim - 1
    conv_filter = ndi.generate_binary_structure(spatialdims, 1).astype(image.dtype)
    conv_filter.ravel()[conv_filter.size // 2] = 0
    conv_filter /= conv_filter.sum()
    if multichannel:
        interp = np.zeros_like(image)
        for i in range(image.shape[-1]):
            interp[..., i] = ndi.convolve(image[..., i], conv_filter, mode='mirror')
    else:
        interp = ndi.convolve(image, conv_filter, mode='mirror')
    return interp