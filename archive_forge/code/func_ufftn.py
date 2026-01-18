import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def ufftn(inarray, dim=None):
    """N-dimensional unitary Fourier transform.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.
    dim : int, optional
        The last axis along which to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : ndarray (same shape than inarray)
        The unitary N-D Fourier transform of ``inarray``.

    Examples
    --------
    >>> input = np.ones((3, 3, 3))
    >>> output = ufftn(input)
    >>> np.allclose(np.sum(input) / np.sqrt(input.size), output[0, 0, 0])
    True
    >>> output.shape
    (3, 3, 3)
    """
    if dim is None:
        dim = inarray.ndim
    outarray = fft.fftn(inarray, axes=range(-dim, 0), norm='ortho')
    return outarray