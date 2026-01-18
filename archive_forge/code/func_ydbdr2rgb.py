from warnings import warn
import numpy as np
from scipy import linalg
from .._shared.utils import (
from ..util import dtype, dtype_limits
@channel_as_last_axis()
def ydbdr2rgb(ydbdr, *, channel_axis=-1):
    """YDbDr to RGB color space conversion.

    Parameters
    ----------
    ydbdr : (..., C=3, ...) array_like
        The image in YDbDr format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `ydbdr` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    This is the color space commonly used by video codecs, also called the
    reversible color transform in JPEG2000.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YDbDr
    """
    return _convert(rgb_from_ydbdr, ydbdr)