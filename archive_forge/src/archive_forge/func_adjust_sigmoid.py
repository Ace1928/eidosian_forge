import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image.

    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.

    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool, optional
        If True, returns the negative sigmoid correction. Defaults to False.

    Returns
    -------
    out : ndarray
        Sigmoid corrected output image.

    See Also
    --------
    adjust_gamma

    References
    ----------
    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast
           Enhancement Functions",
           http://markfairchild.org/PDFs/PAP07.pdf

    """
    _assert_non_negative(image)
    dtype = image.dtype.type
    scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
    if inv:
        out = (1 - 1 / (1 + np.exp(gain * (cutoff - image / scale)))) * scale
        return dtype(out)
    out = 1 / (1 + np.exp(gain * (cutoff - image / scale))) * scale
    return out.astype(dtype)