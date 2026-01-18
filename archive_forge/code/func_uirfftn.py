import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
def uirfftn(inarray, dim=None, shape=None):
    """N-dimensional inverse real unitary Fourier transform.

    This transform considers the Hermitian property of the transform
    from complex to real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.
    dim : int, optional
        The last axis along which to compute the transform. All
        axes by default.
    shape : tuple of int, optional
        The shape of the output. The shape of ``rfft`` is ambiguous in
        case of odd-valued input shape. In this case, this parameter
        should be provided. See ``np.fft.irfftn``.

    Returns
    -------
    outarray : ndarray
        The unitary N-D inverse real Fourier transform of ``inarray``.

    Notes
    -----
    The ``uirfft`` function assumes that the output array is
    real-valued. Consequently, the input is assumed to have a Hermitian
    property and redundant values are implicit.

    Examples
    --------
    >>> input = np.ones((5, 5, 5))
    >>> output = uirfftn(urfftn(input), shape=input.shape)
    >>> np.allclose(input, output)
    True
    >>> output.shape
    (5, 5, 5)
    """
    if dim is None:
        dim = inarray.ndim
    outarray = fft.irfftn(inarray, shape, axes=range(-dim, 0), norm='ortho')
    return outarray