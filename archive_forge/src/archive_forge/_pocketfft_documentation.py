import functools
from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides

    Computes the inverse of `rfft2`.

    Parameters
    ----------
    a : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        .. versionadded:: 1.10.0

        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The forward two-dimensional FFT of real input,
            of which `irfft2` is the inverse.
    rfft : The one-dimensional FFT for real input.
    irfft : The inverse of the one-dimensional FFT of real input.
    irfftn : Compute the inverse of the N-dimensional FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    Examples
    --------
    >>> a = np.mgrid[:5, :5][0]
    >>> A = np.fft.rfft2(a)
    >>> np.fft.irfft2(A, s=a.shape)
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])
    