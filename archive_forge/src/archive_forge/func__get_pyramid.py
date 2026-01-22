import numpy as np
from scipy import ndimage as ndi
from ..transform import pyramid_reduce
from ..util.dtype import _convert
def _get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    """Construct image pyramid.

    Parameters
    ----------
    I : ndarray
        The image to be preprocessed (Grayscale or RGB).
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    pyramid : list[ndarray]
        The coarse to fine images pyramid.

    """
    pyramid = [I]
    size = min(I.shape)
    count = 1
    while count < nlevel and size > downscale * min_size:
        J = pyramid_reduce(pyramid[-1], downscale, channel_axis=None)
        pyramid.append(J)
        size = min(J.shape)
        count += 1
    return pyramid[::-1]