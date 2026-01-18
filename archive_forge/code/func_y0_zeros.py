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
def y0_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

    The derivatives are given by Y0'(z0) = -Y1(z0) at each zero z0.

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
    z0n : ndarray
        Location of nth zero of Y0(z)
    y0pz0n : ndarray
        Value of derivative Y0'(z0) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first 4 real roots and the derivatives at the roots of
    :math:`Y_0`:

    >>> import numpy as np
    >>> from scipy.special import y0_zeros
    >>> zeros, grads = y0_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Roots: {zeros}")
    ...     print(f"Gradients: {grads}")
    Roots: [ 0.89358+0.j  3.95768+0.j  7.08605+0.j 10.22235+0.j]
    Gradients: [-0.87942+0.j  0.40254+0.j -0.3001 +0.j  0.2497 +0.j]

    Plot the real part of :math:`Y_0` and the first four computed roots.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y0
    >>> xmin = 0
    >>> xmax = 11
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.hlines(0, xmin, xmax, color='k')
    >>> ax.plot(x, y0(x), label=r'$Y_0$')
    >>> zeros, grads = y0_zeros(4)
    >>> ax.scatter(zeros.real, np.zeros((4, )), s=30, c='r',
    ...            label=r'$Y_0$_zeros', zorder=5)
    >>> ax.set_ylim(-0.5, 0.6)
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend(ncol=2)
    >>> plt.show()

    Compute the first 4 complex roots and the derivatives at the roots of
    :math:`Y_0` by setting ``complex=True``:

    >>> y0_zeros(4, True)
    (array([ -2.40301663+0.53988231j,  -5.5198767 +0.54718001j,
             -8.6536724 +0.54841207j, -11.79151203+0.54881912j]),
     array([ 0.10074769-0.88196771j, -0.02924642+0.5871695j ,
             0.01490806-0.46945875j, -0.00937368+0.40230454j]))
    """
    if not isscalar(nt) or floor(nt) != nt or nt <= 0:
        raise ValueError('Arguments must be scalar positive integer.')
    kf = 0
    kc = not complex
    return _specfun.cyzo(nt, kf, kc)