import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def y1p_zeros(nt, complex=False):
    """Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

    The values are given by Y1(z1) at each z1 where Y1'(z1)=0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1pn : ndarray
        Location of nth zero of Y1'(z)
    y1z1pn : ndarray
        Value of derivative Y1(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`Y_1'` and the values of
    :math:`Y_1` at these roots.

    >>> import numpy as np
    >>> from scipy.special import y1p_zeros
    >>> y1grad_roots, y1_values = y1p_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Y1' Roots: {y1grad_roots}")
    ...     print(f"Y1 values: {y1_values}")
    Y1' Roots: [ 3.68302+0.j  6.9415 +0.j 10.1234 +0.j 13.28576+0.j]
    Y1 values: [ 0.41673+0.j -0.30317+0.j  0.25091+0.j -0.21897+0.j]

    `y1p_zeros` can be used to calculate the extremal points of :math:`Y_1`
    directly. Here we plot :math:`Y_1` and the first four extrema.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y1, yvp
    >>> y1_roots, y1_values_at_roots = y1p_zeros(4)
    >>> real_roots = y1_roots.real
    >>> xmax = 15
    >>> x = np.linspace(0, xmax, 500)
    >>> x[0] += 1e-15
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y1(x), label=r'$Y_1$')
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$")
    >>> ax.scatter(real_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $Y_1'$", zorder=5)
    >>> ax.scatter(real_roots, y1_values_at_roots.real, s=30, c='k',
    ...            label=r"Extrema of $Y_1$", zorder=5)
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.5, 0.5)
    >>> ax.set_xlim(0, xmax)
    >>> ax.legend(ncol=2, bbox_to_anchor=(1., 0.75))
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if not isscalar(nt) or floor(nt) != nt or nt <= 0:
        raise ValueError('Arguments must be scalar positive integer.')
    kf = 2
    kc = not complex
    return _specfun.cyzo(nt, kf, kc)