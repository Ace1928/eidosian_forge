from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1):
    """XYZ to CIE-Luv color space conversion.

    Parameters
    ----------
    xyz : (..., C=3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-Luv format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., C=3, ...).
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    Notes
    -----
    By default XYZ conversion weights use observer=2A. Reference whitepoint
    for D65 Illuminant, with XYZ tristimulus values of ``(95.047, 100.,
    108.883)``. See function :func:`~.xyz_tristimulus_values` for a list of supported
    illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELUV

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2luv
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_luv = xyz2luv(img_xyz)
    """
    input_is_one_pixel = xyz.ndim == 1
    if input_is_one_pixel:
        xyz = xyz[np.newaxis, ...]
    arr = _prepare_colorarray(xyz, channel_axis=-1)
    x, y, z = (arr[..., 0], arr[..., 1], arr[..., 2])
    eps = np.finfo(arr.dtype).eps
    xyz_ref_white = xyz_tristimulus_values(illuminant=illuminant, observer=observer, dtype=arr.dtype)
    L = y / xyz_ref_white[1]
    mask = L > 0.008856
    L[mask] = 116.0 * np.cbrt(L[mask]) - 16.0
    L[~mask] = 903.3 * L[~mask]
    uv_weights = np.array([1, 15, 3], dtype=arr.dtype)
    u0 = 4 * xyz_ref_white[0] / (uv_weights @ xyz_ref_white)
    v0 = 9 * xyz_ref_white[1] / (uv_weights @ xyz_ref_white)

    def fu(X, Y, Z):
        return 4.0 * X / (X + 15.0 * Y + 3.0 * Z + eps)

    def fv(X, Y, Z):
        return 9.0 * Y / (X + 15.0 * Y + 3.0 * Z + eps)
    u = 13.0 * L * (fu(x, y, z) - u0)
    v = 13.0 * L * (fv(x, y, z) - v0)
    out = np.stack([L, u, v], axis=-1)
    if input_is_one_pixel:
        out = np.squeeze(out, axis=0)
    return out