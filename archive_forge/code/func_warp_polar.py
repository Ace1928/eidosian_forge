import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
@channel_as_last_axis()
def warp_polar(image, center=None, *, radius=None, output_shape=None, scaling='linear', channel_axis=None, **kwargs):
    """Remap image to polar or log-polar coordinates space.

    Parameters
    ----------
    image : (M, N[, C]) ndarray
        Input image. For multichannel images `channel_axis` has to be specified.
    center : 2-tuple, optional
        `(row, col)` coordinates of the point in `image` that represents the center of
        the transformation (i.e., the origin in Cartesian space). Values can be of
        type `float`. If no value is given, the center is assumed to be the center point
        of `image`.
    radius : float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape : tuple (row, col), optional
    scaling : {'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    **kwargs : keyword arguments
        Passed to `transform.warp`.

    Returns
    -------
    warped : ndarray
        The polar or log-polar warped image.

    Examples
    --------
    Perform a basic polar warp on a grayscale image:

    >>> from skimage import data
    >>> from skimage.transform import warp_polar
    >>> image = data.checkerboard()
    >>> warped = warp_polar(image)

    Perform a log-polar warp on a grayscale image:

    >>> warped = warp_polar(image, scaling='log')

    Perform a log-polar warp on a grayscale image while specifying center,
    radius, and output shape:

    >>> warped = warp_polar(image, (100,100), radius=100,
    ...                     output_shape=image.shape, scaling='log')

    Perform a log-polar warp on a color image:

    >>> image = data.astronaut()
    >>> warped = warp_polar(image, scaling='log', channel_axis=-1)
    """
    multichannel = channel_axis is not None
    if image.ndim != 2 and (not multichannel):
        raise ValueError(f'Input array must be 2-dimensional when `channel_axis=None`, got {image.ndim}')
    if image.ndim != 3 and multichannel:
        raise ValueError(f'Input array must be 3-dimensional when `channel_axis` is specified, got {image.ndim}')
    if center is None:
        center = np.array(image.shape)[:2] / 2 - 0.5
    if radius is None:
        w, h = np.array(image.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)
    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]
    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / np.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")
    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}
    warped = warp(image, map_func, map_args=warp_args, output_shape=output_shape, **kwargs)
    return warped