import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type
Return the transfer function of the Laplacian.

    Laplacian is the second order difference, on row and column.

    Parameters
    ----------
    ndim : int
        The dimension of the Laplacian.
    shape : tuple
        The support on which to compute the transfer function.
    is_real : boolean, optional
       If True (default), imp_resp is assumed to be real-valued and
       the Hermitian property is used with rfftn Fourier transform
       to return the transfer function.

    Returns
    -------
    tf : array_like, complex
        The transfer function.
    impr : array_like, real
        The Laplacian.

    Examples
    --------
    >>> tf, ir = laplacian(2, (32, 32))
    >>> np.all(ir == np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    True
    >>> np.all(tf == ir2tf(ir, (32, 32)))
    True
    