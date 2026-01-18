from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
def tilbert(x, h, period=None, _cache=_cache):
    """
    Return h-Tilbert transform of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

        y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
        y_0 = 0

    Parameters
    ----------
    x : array_like
        The input array to transform.
    h : float
        Defines the parameter of the Tilbert transform.
    period : float, optional
        The assumed period of the sequence. Default period is ``2*pi``.

    Returns
    -------
    tilbert : ndarray
        The result of the transform.

    Notes
    -----
    If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd, then
    ``tilbert(itilbert(x)) == x``.

    If ``2 * pi * h / period`` is approximately 10 or larger, then
    numerically ``tilbert == hilbert``
    (theoretically oo-Tilbert == Hilbert).

    For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.

    """
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return tilbert(tmp.real, h, period) + 1j * tilbert(tmp.imag, h, period)
    if period is not None:
        h = h * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, h=h):
            if k:
                return 1.0 / tanh(h * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, h] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)